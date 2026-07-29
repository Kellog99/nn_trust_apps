from logging import Logger
from pathlib import Path
from typing import Iterator, Optional, Any, Callable

import ray
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmarking.utils import evaluate_attack
from benchmarking.utils.evaluation import save_attack_result
from models import ModelReportProps, JobExecutionConfig
from models.benchmark import AttackEvaluation, JobResult
from nn_trust import StatisticComposer, ModelAdapter


class BenchmarkExecutor:
    """
    Executes a list of benchmark jobs either locally (serial) or distributed via Ray.
    """

    def __init__(
            self,
            benchmark_id: str,
            root_path: str | Path = None,
            verbose: bool = False,
            use_ray: bool = False,
            num_gpus_per_job: float = 0.4,
    ):
        """
        Args:
            root_path: Base directory where benchmark results are saved. Accepts
                either a `str` (which is expanded, e.g. resolving `~`) or an
                already-constructed `Path`. Left as `None` by default, so callers
                relying on the default must set it before results can be saved
                (`save_attack_result` and the final `output_path` both depend on it).
            verbose: When `True`, enables progress bars (`tqdm`), a start-of-run
                print in `__call__`, an end-of-run success/failure summary, and
                a default module logger inside `execute_jobs` if no `log` is
                explicitly passed in.
            use_ray: Selects the execution backend. `True` distributes jobs across
                a Ray cluster (`_iter_ray`, with jobs submitted concurrently and
                collected as they complete); `False` runs jobs one at a time on
                the local process (`_iter_local`).
            num_gpus_per_job: Fraction (or whole number) of a GPU to reserve per
                job when `use_ray=True`. E.g. `0.4` lets ~2 jobs share one GPU.
                Only used to configure the Ray remote wrapper; ignored when
                `use_ray=False`.


        Attributes set:
            self.root_path: See `root_path` above (normalized to a `Path`).
            self.verbose: See `verbose` above.
            self.use_ray: See `use_ray` above.
            self._remote_execute_job: The Ray remote-wrapped version of
                `execute_job`, created only when `use_ray=True`; `None` otherwise.
        """
        self.benchmark_id = benchmark_id
        self.root_path = Path(root_path).expanduser() if isinstance(root_path, str) else root_path
        self.verbose = verbose
        self.use_ray = use_ray
        self._remote_execute_job = ray.remote(num_gpus=num_gpus_per_job)(execute_job) if use_ray else None

    def save_results(self, job_output: ModelReportProps) -> None:
        save_attack_result(
            benchmark_id=job_output["benchmark_job_info"]["benchmark_id"],
            atk_result=job_output["attack_results"],
            atk_id=job_output["benchmark_job_info"]["atk_id"],
            dataset_name=job_output["benchmark_job_info"]["dataset_id"],
            model_name=job_output["benchmark_job_info"]["model_id"],
            root_path=self.root_path,
        )

    @staticmethod
    def _describe(job_config: JobExecutionConfig) -> str:
        return (
            f"model={getattr(job_config.model, 'model_id', job_config.model)} "
            f"dataset={getattr(job_config.dataset, 'dataset_id', job_config.dataset)} "
            f"attack={getattr(job_config.attack, 'atk_id', job_config.attack)}"
        )

    # ------------------------------------------------------------------ #
    # Execution backends — both yield JobResult(job_config, result, error)
    # ------------------------------------------------------------------ #

    def _iter_local(
            self,
            model: ModelAdapter,
            dataloader: DataLoader,
            attacks: list[dict[str, Any]],
            statistics: StatisticComposer,
            device: torch.device = torch.device("cpu"),
            log: Optional[Logger] = None,
    ) -> Iterator[JobResult]:
        """
        Execute all the attacks locally in a serialized way.
        """
        pbar = tqdm(attacks, disable=not self.verbose, desc="Running jobs")
        for atk in pbar:
            result = evaluate_attack(
                dataloader=dataloader,
                model=model,
                attack=atk,
                statistics=statistics,
                device=device,
            )
            yield JobResult(job_config=atk, result=result, error=None)

    def _iter_ray(
            self,
            model: ModelAdapter,
            dataloader: DataLoader,
            attacks: list[dict[str, Any]],
            statistics: StatisticComposer,
            device: torch.device = torch.device("cpu"),
            log: Optional[Logger] = None,
    ) -> Iterator[JobResult]:
        pending = {
            self._remote_execute_job.remote(
                dataloader=dataloader,
                model=model,
                attack=atk,
                statistics=statistics,
                device=device,
            ): atk
            for atk in attacks
        }

        pbar = tqdm(total=len(pending), disable=not self.verbose, desc="Running jobs")
        with pbar:
            while pending:
                done, _ = ray.wait(list(pending.keys()), num_returns=1)
                ref = done[0]
                atk = pending.pop(ref)
                pbar.update(1)
                try:
                    result = ray.get(ref)  # execute_job must return the evaluate_attack output
                    yield JobResult(job_config=atk, result=result, error=None)
                except Exception as e:
                    yield JobResult(job_config=atk, result=None, error=e)

    def execute_jobs(
            self,
            model: ModelAdapter,
            dataloader: DataLoader,
            attacks: list[dict[str, Any]],
            statistics: StatisticComposer,
            device: torch.device = torch.device("cpu"),
            log: Optional[Logger] = None,
            verbose: bool = True
    ) -> dict:
        func: Callable[..., Iterator[JobResult]] = self._iter_ray if self.use_ray else self._iter_local
        results_iter: Iterator[JobResult] = func(
            model=model,
            dataloader=dataloader,
            attacks=attacks,
            statistics=statistics,
            device=device,
            log=log,
            verbose=verbose
        )

        results: dict[str, Any] = {}
        succeeded: list[JobResult] = []
        failed: list[JobResult] = []

        for job_result in results_iter:
            atk_id = job_result.job_config.get("id") or job_result.job_config.get("name")
            if job_result.error is None:
                self.save_results(job_result.result)
                results[atk_id] = job_result.result
                succeeded.append(job_result)
            else:
                failed.append(job_result)
                if log is not None:
                    log.error(f"Job failed: {job_result.job_config}: {job_result.error}")

        return {
            "output_path": Path(self.root_path) / self.benchmark_id,
            "results": results,
            "succeeded": succeeded,
            "failed": failed,
        }

    def __repr__(self):
        backend = "ray" if self.use_ray else "local"
        return f"{self.__class__.__name__}(root_path={self.root_path!r}, verbose={self.verbose!r}, backend={backend!r})"

    def __call__(self, input_job_list: list[JobExecutionConfig]) -> dict:
        if self.verbose:
            print(f"Starting execution of {len(input_job_list)} jobs via {self!r}")
        return self.execute_jobs(input_job_list)
