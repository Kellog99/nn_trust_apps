from datetime import datetime
from logging import Logger
from pathlib import Path
from threading import Lock
from typing import Iterator, Optional, Any, Callable

import torch
from torch.utils.data import DataLoader

from benchmarking.utils.execution import _iter_ray, _iter_local
from models import JobResult
from models.reports import ReportAttackProps, AttackMetricsProps, ParameterLog
from nn_trust import StatisticComposer, ModelAdapter


class ProgressTracker:
    """Small in-memory registry used by the jobs API."""

    def __init__(self) -> None:
        self._tasks: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def create_task(
            self,
            attack_id: str,
            benchmark_id: str,
            name: Optional[str] = None,
    ) -> str:
        with self._lock:
            task_id = f"{attack_id}_{benchmark_id}_{len(self._tasks)}"
            self._tasks[task_id] = {
                "benchmark_id": benchmark_id,
                "attack_id": attack_id,
                "name": name or attack_id,
                "status": "created",
                "progress": 0,
                "error": None,
            }
        return task_id

    def update_task(
            self,
            task_id: str,
            status: str,
            progress: Optional[int] = None,
            error: Optional[str] = None,
    ) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task["status"] = status
            if progress is not None:
                task["progress"] = progress
            if error is not None:
                task["error"] = error

    def list_tasks(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {task_id: task.copy() for task_id, task in self._tasks.items()}


tracker = ProgressTracker()


class BenchmarkExecutor:
    """
    Executes a list of benchmark jobs either locally (serial) or distributed via Ray.
    """

    def __init__(
            self,
            benchmark_id: Optional[str] = None,
            verbose: bool = False,
            use_ray: bool = False,
            num_gpus_per_job: float = 0.4,
            output_path: Optional[str | Path] = None,
            device: torch.device = torch.device("cpu"),
    ):
        if benchmark_id is None:
            benchmark_id: str = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.benchmark_id = benchmark_id

        self.verbose = verbose
        self.device = device
        output_path = output_path if output_path is not None else f"./tmp/{self.benchmark_id}"
        if isinstance(output_path, str):
            output_path = Path(output_path).expanduser().resolve()
            output_path.mkdir(parents=True, exist_ok=True)
        self.output_path: Path = output_path

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
            max_saved_elements: int = 10,
    ) -> dict[str, ReportAttackProps]:

        task_ids: dict[str, str] = {}
        for attack in attacks:
            attack_id = str(attack.get("id") or attack.get("name", "unknown"))
            task_id = tracker.create_task(
                attack_id=attack_id,
                benchmark_id=self.benchmark_id,
                name=attack.get("name"),
            )
            task_ids[attack_id] = task_id
            tracker.update_task(
                task_id,
                status="in_progress",
            )

        func: Callable[..., Iterator[JobResult]] = _iter_ray if self.use_ray else _iter_local
        results_iter: Iterator[JobResult] = func(
            model=model,
            dataloader=dataloader,
            attacks=attacks,
            statistics=statistics,
            device=device if device is not None else self.device,
            max_saved_elements=max_saved_elements,
            output_path=self.output_path,
            log=log,
        )

        results: dict[str, ReportAttackProps] = {}
        failed: list[JobResult] = []

        try:
            for jr in results_iter:
                task_id = task_ids.get(jr.id)

                if jr.error is None and jr.result is not None:
                    params: list[ParameterLog] | None = jr.parameters
                    # removing all the unnecessary elements
                    if params is None:
                        raise ValueError("The list of parameters is None.")
                    filtered_params: list[ParameterLog] = [
                        param for param in params if param.id != "model"
                    ]
                    results[jr.id] = ReportAttackProps(
                        name=jr.id,
                        parameters=filtered_params,
                        metrics=AttackMetricsProps.model_validate(jr.result)
                    )
                    if task_id is not None:
                        tracker.update_task(
                            task_id,
                            status="completed",
                            progress=100,
                        )
                else:
                    failed.append(jr)
                    if task_id is not None:
                        tracker.update_task(
                            task_id,
                            status="failed",
                            progress=100,
                            error=str(jr.error or "Attack returned no result"),
                        )
                    if log is not None:
                        log.error(f"Job failed: {jr.id}: {jr.error}")
        except Exception as exc:
            current_tasks = tracker.list_tasks()
            for task_id in task_ids.values():
                if current_tasks[task_id]["status"] == "in_progress":
                    tracker.update_task(
                        task_id,
                        status="failed",
                        progress=100,
                        error=str(exc),
                    )
            raise

        if failed:
            details = "; ".join(f"{job.id}: {job.error}" for job in failed)
            raise RuntimeError(f"Benchmark job(s) failed: {details}")

        return results

    def __repr__(self):
        backend = "ray" if self.use_ray else "local"
        return f"{self.__class__.__name__}(output_path={self.output_path!r}, verbose={self.verbose!r}, backend={backend!r})"
