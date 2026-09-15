import json
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
from nn_trust import StatisticComposer, ModelAdapter


class BenchmarkExecutor:
    """
    Executes a list of benchmark jobs either locally (serial) or distributed via Ray.
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

    def _save_attack_result(
            self,
            attack_id: str,
            result: ReportAttackProps | None = None,
            error: str | None = None,
    ) -> None:
        """Persist the latest result for one attack in its artifact directory."""
        attack_path = self.output_path / attack_id
        attack_path.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {"id": attack_id}
        if result is not None:
            payload.update(result.model_dump(mode="json"))
        if error is not None:
            payload["status"] = "error"
            payload["error"] = error

        with (attack_path / "job_results.json").open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

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
        pbar = tqdm(attacks.items())

        for id, params in pbar:
            try:
                yield evaluate_attack(
                    dataloader=dataloader,
                    model=model,
                    attack_id=id,
                    parameters=params,
                    statistics=statistics,
                    device=device,
                    verbose=False,
                    output_path=output_path,
                    max_saved_elements=max_saved_elements if max_saved_elements is not None else 10,
                )
            except Exception as exc:  # noqa: BLE001 - intentional: isolate per-attack failures
                if log is not None:
                    log.exception(f"Attack '{id}' failed")
                yield JobResult(
                    id=id,
                    result=None,
                    parameters=None,
                    status="error",
                    error=str(exc),
                )

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
                attack_result = ReportAttackProps(
                    name=jr.id,
                    parameters=filtered_params,
                    metrics=AttackMetricsProps.model_validate(jr.result),
                )
                results[jr.id] = attack_result
            else:
                failed.append(jr)
                error = jr.error or "Attack returned no result"
                self._save_attack_result(
                    jr.id,
                    error=error,
                )
                if log is not None:
                    log.error(f"Job failed: {jr.id}: {error}")

        if failed:
            details = "; ".join(
                f"{job.id}: {job.error or job.status}"
                for job in failed
            )
            raise RuntimeError(f"Benchmark job(s) failed: {details}")

        return results
