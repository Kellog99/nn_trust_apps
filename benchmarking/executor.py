from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Iterator, Optional, Any, Callable

import torch
from torch.utils.data import DataLoader

from benchmarking.utils.execution import _iter_ray, _iter_local
from models import JobResult
from models.reports import ReportAttackProps, AttackMetricsProps, ParameterLog
from nn_trust import StatisticComposer, ModelAdapter


class BenchmarkExecutor:
    """
    Executes a list of benchmark jobs either locally (serial) or distributed via Ray.
    """

    def __init__(
            self,
            benchmark_id: Optional[str] = None,
            root_path: Optional[str | Path] = None,
            verbose: bool = False,
            use_ray: bool = False,
            num_gpus_per_job: float = 0.4,
            output_path: Optional[str | Path] = None,
            device: torch.device = torch.device("cpu"),
    ):
        if benchmark_id is None:
            benchmark_id: str = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.benchmark_id = benchmark_id
        self.root_path = Path(root_path).expanduser() if isinstance(root_path, str) else root_path

        self.verbose = verbose
        self.device = device
        self.output_path = output_path if output_path is not None else f"./tmp/{self.benchmark_id}"
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path).expanduser().resolve()
            self.output_path.mkdir(parents=True, exist_ok=True)

        self.use_ray = use_ray
        self.num_gpus_per_job = num_gpus_per_job

    def execute_jobs(
            self,
            model: ModelAdapter,
            dataloader: DataLoader,
            attacks: list[dict[str, Any]],
            statistics: StatisticComposer,
            log: Optional[Logger] = None,
            device: Optional[torch.device] = None,
            output_path: Optional[str | Path] = None,
            save_variables: Optional[list[str]] = None,
            max_saved_elements: Optional[int | dict[str, int]] = None,
    ) -> dict[str, ReportAttackProps]:
        func: Callable[..., Iterator[JobResult]] = _iter_ray if self.use_ray else _iter_local
        results_iter: Iterator[JobResult] = func(
            model=model,
            dataloader=dataloader,
            attacks=attacks,
            statistics=statistics,
            device=device if device is not None else self.device,
            output_path=output_path if output_path is not None else self.output_path,
            save_variables=save_variables,
            max_saved_elements=max_saved_elements,
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
