from logging import Logger
from pathlib import Path
from typing import Iterator, Optional, Any, Callable

import ray
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmarking.utils import evaluate_attack
from models import JobResult
from models.reports import ReportAttackProps, AttackMetricsProps, ParameterLog
from nn_trust import StatisticComposer, ModelAdapter


class BenchmarkExecutor:
    """
    Executes a list of benchmark jobs either locally (serial) or distributed via Ray.
    """

    def __init__(
            self,
            benchmark_id: str,
            root_path: Optional[str | Path] = None,
            verbose: bool = False,
            use_ray: bool = False,
            num_gpus_per_job: float = 0.4,
    ):
        self.benchmark_id = benchmark_id
        self.root_path = Path(root_path).expanduser() if isinstance(root_path, str) else root_path
        self.verbose = verbose
        self.use_ray = use_ray
        self._remote_execute_job = ray.remote(num_gpus=num_gpus_per_job)(evaluate_attack) if use_ray else None

    @staticmethod
    def _job_id(atk: dict[str, Any]) -> str:
        return atk.get("id") or atk.get("Name") or "atk"

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
            atk_id = self._job_id(atk)

            pbar.desc = f"Running {atk_id}"
            try:
                yield evaluate_attack(
                    dataloader=dataloader,
                    model=model,
                    attack=atk,
                    statistics=statistics,
                    device=device,
                )
            except Exception as e:
                print(f"error {e}")
                yield JobResult(
                    id=atk_id,
                    error=e
                )

    def _iter_ray(
            self,
            model: ModelAdapter,
            dataloader: DataLoader,
            attacks: list[dict[str, Any]],
            statistics: StatisticComposer,
            device: torch.device = torch.device("cpu"),
            log: Optional[Logger] = None,
    ) -> Iterator[JobResult]:
        pending: dict[ray.ObjectRef, dict[str, Any]] = {
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
                atk_id = self._job_id(atk)
                pbar.update(1)
                try:
                    yield ray.get(ref)
                except Exception as e:
                    yield JobResult(
                        id=atk_id,
                        error=e
                    )

    def execute_jobs(
            self,
            model: ModelAdapter,
            dataloader: DataLoader,
            attacks: list[dict[str, Any]],
            statistics: StatisticComposer,
            device: torch.device = torch.device("cpu"),
            log: Optional[Logger] = None,
    ) -> dict[str, ReportAttackProps]:
        func: Callable[..., Iterator[JobResult]] = self._iter_ray if self.use_ray else self._iter_local
        results_iter: Iterator[JobResult] = func(
            model=model,
            dataloader=dataloader,
            attacks=attacks,
            statistics=statistics,
            device=device,
            log=log,
        )

        results: dict[str, ReportAttackProps] = {}
        failed: list[JobResult] = []

        for jr in results_iter:
            if jr.error is None and jr.result is not None:
                params: list[ParameterLog] | None = jr.parameters
                # removing all the unnecessary elements
                if params is not None:
                    params: list[ParameterLog] = [param for param in params if param.id != "model"]
                    results[jr.id] = ReportAttackProps(
                        name=jr.id,
                        parameters=params,
                        metrics=AttackMetricsProps.model_validate(jr.result)
                    )
                else:
                    raise ValueError("The list of parameters is None.")
            else:
                failed.append(jr)
                if log is not None:
                    log.error(f"Job failed: {jr.id}: {jr.error}")

        return results

    def __repr__(self):
        backend = "ray" if self.use_ray else "local"
        return f"{self.__class__.__name__}(root_path={self.root_path!r}, verbose={self.verbose!r}, backend={backend!r})"
