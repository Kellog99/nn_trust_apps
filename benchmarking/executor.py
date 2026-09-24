from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Iterator, Optional, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmarking.utils import evaluate_attack
from models import JobResult
from models.reports import ReportAttackProps, AttackMetricsProps, ParameterLog
from nn_trust import StatisticComposer, ModelAdapter, Task


class BenchmarkExecutor:
    """
    Executes a list of benchmark jobs locally, in order.
    """

    def __init__(
            self,
            benchmark_id: Optional[str] = None,
            verbose: bool = False,
            output_path: Optional[str | Path] = None,
            device: torch.device = torch.device("cpu"),
    ):
        if benchmark_id is None:
            benchmark_id: str = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.benchmark_id: str = benchmark_id

        self.verbose = verbose
        self.device = device
        if output_path is None:
            output_path = f"./tmp/{self.benchmark_id}"

        self.output_path = Path(output_path).expanduser().resolve()
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _save_fallback_error(self, attack_id: str, error: str) -> None:
        """
        Fallback persistence for a failed job, used only when `evaluate_attack`
        itself did not have the chance to write a `job_results.json` (e.g. the
        job raised before `JobResult` was even constructed).
        """
        attack_path = self.output_path / attack_id
        result_file = attack_path / "job_results.json"
        if result_file.exists():
            # evaluate_attack already persisted a (richer) error state, don't overwrite it.
            return

        attack_path.mkdir(parents=True, exist_ok=True)
        JobResult(id=attack_id, status="error", error=error).save(result_file)

    def _iter_local(
            self,
            model: ModelAdapter,
            dataloader: DataLoader,
            attacks: dict[str, dict[str, Any]],
            statistics: StatisticComposer,
            device: torch.device,
            output_path: Path,
            max_saved_elements: int = 10,
            log: Optional[Logger] = None,
    ) -> Iterator[JobResult]:
        """
        Serial execution: yields one JobResult per attack, in order.
        """
        if not attacks:
            if log is not None:
                log.warning("'attacks' is empty, no jobs will run")
            return

        for attack_id, params in tqdm(attacks.items()):
            try:
                yield evaluate_attack(
                    dataloader=dataloader,
                    model=model,
                    attack_id=attack_id,
                    parameters=params,
                    statistics=statistics,
                    device=device,
                    verbose=False,
                    output_path=output_path,
                    max_saved_elements=max_saved_elements if max_saved_elements is not None else 10,
                )
            except Exception as exc:  # noqa: BLE001 - intentional: isolate per-attack failures
                if log is not None:
                    log.exception(f"Attack '{attack_id}' failed")
                yield JobResult(id=attack_id, status="error", error=str(exc))

    def execute_jobs(
            self,
            model: ModelAdapter,
            dataloader: DataLoader,
            attacks: dict[str, dict[str, Any]] | list[dict[str, Any]],
            statistics: StatisticComposer,
            log: Optional[Logger] = None,
            device: Optional[torch.device] = None,
            max_saved_elements: int = 10,
    ) -> dict[str, ReportAttackProps]:

        attack_specs = (
            attacks
            if isinstance(attacks, dict)
            else {
                str(attack["id"]): {
                    key: value for key, value in attack.items() if key != "id"
                }
                for attack in attacks
            }
        )

        results_iter: Iterator[JobResult] = self._iter_local(
            model=model,
            dataloader=dataloader,
            attacks=attack_specs,
            statistics=statistics,
            device=device if device is not None else self.device,
            max_saved_elements=max_saved_elements,
            output_path=self.output_path,
            log=log,
        )

        results: dict[str, ReportAttackProps] = {}
        failed: list[JobResult] = []

        for jr in results_iter:
            if jr.status == "finished" and jr.result is not None:
                params: list[ParameterLog] | None = jr.parameters
                if params is None:
                    raise ValueError("The list of parameters is None.")
                filtered_params = [param for param in params if param.id != "model"]
                results[jr.id] = ReportAttackProps(
                    name=jr.id,
                    parameters=filtered_params,
                    metrics=AttackMetricsProps.model_validate(jr.result),
                )
            else:
                failed.append(jr)
                error = jr.error or "Attack returned no result"
                self._save_fallback_error(jr.id, error)
                if log is not None:
                    log.error(f"Job failed: {jr.id}: {error}")

        if failed:
            details = "; ".join(f"{job.id}: {job.error or job.status}" for job in failed)
            raise RuntimeError(f"Benchmark job(s) failed: {details}")

        return results
