from datetime import datetime
from pathlib import Path
from typing import Any

import time
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from benchmarking.utils.advyolo_evaluation import evaluate_frozen_advyolo, detection_predictions
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
        logger = PyTorchCheckpointLogger(
            path=output_path,
            max_artifact={
                "original_input": max_saved_elements or 1,
                "adversarial_input": max_saved_elements or 1,
            },
        )
        execution_start = time.perf_counter()
        if attack_id == "advyoloevasion":
            attack.generate(gen_train=dataloader)
            statistics = evaluate_frozen_advyolo(
                dataloader=dataloader,
                model=model,
                attack=attack,
                statistics=statistics,
                logger=logger,
                device=device,
                output_path=output_path,
            )
            job_result.progress = job_result.total
        else:
            total_iterations = len(dataloader)
            pbar = enumerate(tqdm(dataloader, desc=f"Attack {attack!r}") if verbose else dataloader, start=1)
            for completed_iterations, (batch, label) in pbar:
                iteration_start = time.perf_counter()

                if not isinstance(batch, torch.Tensor):
                    batch = torch.stack(batch)
                batch = batch.to(device)
                label = to_device(label, device)

                with torch.no_grad():
                    out = model(batch)
                target = (
                    torch.nn.functional.one_hot(label.long(), num_classes=out.shape[-1]).to(out)
                    if task == Task.Classification else out
                )
                x_adv = attack.generate(x=batch, y=target).detach()
                with torch.no_grad():
                    out_adv = model(x_adv)

                if task == Task.Classification:
                    # The identity baseline measures accuracy against ground truth.
                    y_pred = label if attack_id == "identitybaseline" else out.argmax(dim=-1)
                    y_pred_adv = out_adv.argmax(dim=-1)
                    y_target = label
                else:
                    y_pred = detection_predictions(
                        output=out,
                        iou_threshold=attack.config.iou_threshold_evaluation,
                        score_threshold=attack.config.score_threshold_evaluation
                    )
                    y_pred_adv = detection_predictions(
                        output=out_adv,
                        iou_threshold=attack.config.iou_threshold_evaluation,
                        score_threshold=attack.config.score_threshold_evaluation
                    )
                    y_target = y_pred
                    if attack.config.targeted:
                        label_target = attack.config.label_target
                        target_class = (label_target + 1) % out[1].shape[-1]
                        y_target = [
                            {
                                "boxes": pred["boxes"],
                                "labels": torch.where(
                                    pred["labels"] == label_target,
                                    target_class, pred["labels"],
                                ),
                            }
                            for pred in y_pred
                        ]

                for original, adversarial in zip(batch, x_adv):
                    logger.log(tag="original_input", data=original)
                    logger.log(tag="adversarial_input", data=adversarial)
                statistics.update(
                    x=batch.detach(),
                    x_adv=x_adv,
                    y=label,
                    y_target=y_target,
                    y_pred=y_pred,
                    y_pred_adv=y_pred_adv,
                    out=out,
                    out_adv=out_adv,
                )

                job_result.progress += len(batch)
                job_result.iteration_time = time.perf_counter() - iteration_start
                job_result.execution_time = time.perf_counter() - execution_start
                job_result.estimated_execution_time = (
                        job_result.execution_time / completed_iterations * total_iterations
                )
                job_result.save(res_path)

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
