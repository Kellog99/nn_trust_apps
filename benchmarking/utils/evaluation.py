from datetime import datetime
from pathlib import Path
from typing import Any

import time
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from benchmarking.utils.evaluation_pipeline import run_evaluation_pipeline
from models import JobResult
from models.reports import ParameterLog
from nn_trust import AttackFactory, ModelAdapter, StatisticComposer, Task, EvasionAttack
from nn_trust.utils import PyTorchCheckpointLogger, to_device


def evaluate_attack(
        dataloader: DataLoader,
        model: ModelAdapter,
        statistics: StatisticComposer,
        attack_id: str,
        parameters: dict[str, Any] | None = None,
        device: torch.device = torch.device("cpu"),
        verbose: bool = False,
        output_path: str | Path | None = None,
        max_saved_elements: int | None = 10,
) -> JobResult:
    """Evaluate an attack, save artifacts and progress, and update aggregate metrics.

    Classification uses untargeted, ground-truth-based targets on every sample.
    Detection uses the attack's configured targets; AdvYOLO trains its patch
    before evaluating it. Artifacts are saved under ``output_path / attack_id``.
    ``max_saved_elements`` limits each input artifact; None or zero saves one.
    """
    if output_path is None:
        output_path: Path = Path("tmp") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = Path(output_path).expanduser().resolve() / attack_id
    output_path.mkdir(parents=True, exist_ok=True)
    res_path = output_path / "job_results.json"

    job_result = JobResult(
        id=attack_id,
        progress=0,
        total=len(dataloader.dataset)
    )
    job_result.save(res_path)

    attack = None
    logger = None
    try:
        job_result.status = "in progress"
        job_result.save(res_path)
        task = model.task
        if task not in (Task.Classification, Task.Detection):
            raise NotImplementedError(f"{task} not supported yet.")

        attack: EvasionAttack = AttackFactory.create(
            class_id=attack_id,
            model=model,
            device=device,
            task=task,
            **(parameters or {}),
        )

        for metric_id, metric in statistics._performance_stats.items():
            if metric_id in {"misclassification", "misdetection"}:
                metric.targeted = attack.config.targeted

            if task == Task.Detection and hasattr(metric, "label_target"):
                metric.label_target = attack.config.label_target
        logger = PyTorchCheckpointLogger(
            path=output_path,
            max_artifact={
                "original_input": max_saved_elements or 1,
                "adversarial_input": max_saved_elements or 1,
            },
        )
        execution_start = time.perf_counter()
        def on_batch(completed_iterations, batch_size, iteration_time):
            job_result.progress += batch_size
            job_result.iteration_time = iteration_time
            job_result.execution_time = time.perf_counter() - execution_start
            job_result.estimated_execution_time = (
                job_result.execution_time / completed_iterations * len(dataloader)
            )
            job_result.save(res_path)

        run_evaluation_pipeline(
            model=model, attack=attack, attack_id=attack_id,
            dataloader=dataloader, statistics=statistics, device=device,
            logger=logger, output_path=output_path, on_batch=on_batch, verbose=verbose,
        )
        if attack_id == "advyoloevasion":
            job_result.progress = job_result.total

        result = statistics.compute()
        # AdvYOLO's evaluator already updates the aggregate statistics.
        if attack_id not in ("identitybaseline", "advyoloevasion"):
            statistics.update_aggregate(statistics.get_raw_state())
        statistics.reset()

        attack_parameters = attack.config.model_dump()
        job_result.parameters = [
            ParameterLog(
                id=key,
                name=field.title,
                description=field.description,
                value=attack_parameters[key]
            )
            for key, field in type(attack.config).model_fields.items()
            if key != "model" and isinstance(attack_parameters.get(key), (int, float, bool))
        ]
        job_result.result = result

        job_result.execution_time = time.perf_counter() - execution_start
        job_result.status = "finished"
        job_result.save(res_path)
        return job_result
    except Exception as exc:
        job_result.status = "error"
        job_result.error = str(exc)
        job_result.save(res_path)
        raise
    finally:
        try:
            if logger is not None:
                logger.close()
        finally:
            if attack is not None:
                attack.logger.close()
