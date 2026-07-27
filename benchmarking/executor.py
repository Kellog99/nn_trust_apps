from logging import Logger
from pathlib import Path
from typing import Iterator, Optional

import ray
import torch
from tqdm import tqdm

from benchmarking.utils.evaluation import save_attack_result
from benchmarking.utils.execution import execute_job
from models import ModelReportProps, JobExecutionConfig
from models.benchmark import AttackEvaluation, JobResult


class BenchmarkExecutor:
    """
    Executes a list of benchmark jobs either locally (serial) or distributed via Ray.
    """

    def __init__(
            self,
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
            input_job_list: list[JobExecutionConfig],
            log: Optional[Logger] = None,
    ) -> Iterator[JobResult]:
        """
        Execute all the attacks locally in a serialized way.
        """
        for job_config in tqdm(input_job_list, disable=not self.verbose, desc="Running jobs"):
            try:
                result: AttackEvaluation = execute_job(
                    benchmark_id=job_config.benchmark_id,
                    dataset_cnf=job_config.dataset,
                    model_cnf=job_config.model,
                    attack_cnf=job_config.attack,
                    metrics=job_config.evaluation,
                    options=job_config.options
                )
                yield JobResult(
                    job_config=job_config,
                    result=result,
                    error=None
                )
            except Exception as exc:
                if log:
                    log.exception("Job failed [%s]", self._describe(job_config))
                yield JobResult(
                    job_config=job_config,
                    result=None,
                    error=exc
                )

    def _iter_ray(
            self,
            input_job_list: list[JobExecutionConfig],
            log: Optional[Logger] = None,
    ) -> Iterator[JobResult]:
        pending = {
            self._remote_execute_job.remote(
                benchmark_id=job_config.benchmark_id,
                dataset_cnf=job_config.dataset,
                model_cnf=job_config.model,
                attack=job_config.attack,
                metrics=job_config.evaluation,
                options=job_config.options,
            ): job_config
            for job_config in input_job_list
        }

        with tqdm(total=len(pending), disable=not self.verbose, desc="Running jobs") as pbar:
            while pending:
                done, _ = ray.wait(list(pending.keys()), num_returns=1)
                ref = done[0]
                job_config = pending.pop(ref)
                pbar.update(1)
                try:
                    yield JobResult(
                        job_config=job_config,
                        result=ray.get(ref),
                        error=None
                    )
                except Exception as exc:
                    if log:
                        log.exception("Job failed [%s]", self._describe(job_config))
                    yield JobResult(
                        job_config=job_config,
                        result=None,
                        error=exc
                    )

    def execute_jobs(
            self,
            input_job_list: list[JobExecutionConfig],
            log: Optional[Logger] = None,
    ) -> dict:
        if not input_job_list:
            raise ValueError("input_job_list must not be empty")
        if self.use_ray:
            results_iter: Iterator[JobResult] = self._iter_ray(input_job_list, log=log)
        else:
            results_iter: Iterator[JobResult] = self._iter_local(input_job_list, log=log)

        first_benchmark_id = None
        succeeded, failed = 0, 0
        failures: list[JobExecutionConfig] = []

        for job_config, result, error in results_iter:
            if error is not None:
                print(error)
                failed += 1
                failures.append(error)
                continue

            self.save_results(result)
            succeeded += 1
            if first_benchmark_id is None:
                first_benchmark_id = result.id

        if first_benchmark_id is None:
            # Every single job failed — nothing was ever saved.
            raise RuntimeError(
                f"All {failed} job(s) failed; no results were produced. "
                f"First error: {failures!r}" if failures else "No results produced."
            )

        if self.verbose and failed:
            print(f"Completed with {succeeded} succeeded, {failed} failed.")

        return {
            "output_path": Path(self.root_path) / first_benchmark_id,
            "succeeded": succeeded,
            "failed": failed,
            "failures": failures,
        }

    def __repr__(self):
        backend = "ray" if self.use_ray else "local"
        return f"{self.__class__.__name__}(root_path={self.root_path!r}, verbose={self.verbose!r}, backend={backend!r})"

    def __call__(self, input_job_list: list[JobExecutionConfig]) -> dict:
        if self.verbose:
            print(f"Starting execution of {len(input_job_list)} jobs via {self!r}")
        return self.execute_jobs(input_job_list)
