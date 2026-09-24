from typing import Any, Optional

import torch
import torchvision
from torch.utils.data import DataLoader

from nn_trust import ModelAdapter, EvasionAttack
from nn_trust.attack.utils.detection import nms
from nn_trust.utils import to_device


def detection_predictions(
        output,
        iou_threshold: Optional[float] = None,
        score_threshold: Optional[float] = None
) -> list[dict[str, torch.Tensor]]:
    """
    Run NMS on a raw (boxes, scores) model output.
    """
    if iou_threshold is None:
        raise ValueError("The iou_threshold must be not None.")
    if score_threshold is None:
        raise ValueError("The score_threshold cannot be None.")

    boxes, scores = output
    return nms(
        y={
            "boxes": boxes,
            "scores": scores.max(dim=-1).values,
            "cls_scores": scores
        },
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )


def _log_batch(logger, **tagged_tensors):
    """Log per-sample tensors under their tag, in lockstep across tags."""
    size = len(next(iter(tagged_tensors.values())))
    for i in range(size):
        for tag, values in tagged_tensors.items():
            logger.log(tag=tag, data=values[i])


def evaluate_frozen_advyolo(
        dataloader: DataLoader,
        model: ModelAdapter,
        attack: EvasionAttack,
        statistics,
        logger,
        device: torch.device,
        output_path,
):
    patch = attack.patch.detach().cpu().clone()

    if attack.config.save:
        torch.save({"patch": patch}, output_path / "patch.pt")
        torchvision.utils.save_image(patch.clamp(0, 1), output_path / "patch.png")

    iou_threshold_evaluation: float | int | None = attack.config.iou_threshold_evaluation
    score_threshold_evaluation: float | int | None = attack.config.score_threshold_evaluation
    iou_threshold_optimization: float | int | None = attack.config.iou_threshold_optimization
    score_threshold_optimization: float | int | None = attack.config.score_threshold_optimization

    # Evaluate the final frozen patch
    for batch, label in dataloader:
        batch = batch.to(device)
        label = to_device(label, device)

        with torch.no_grad():
            out = model(batch)

        evaluation_preds = detection_predictions(
            output=out,
            iou_threshold=iou_threshold_evaluation,
            score_threshold=score_threshold_evaluation
        )
        optimization_preds = detection_predictions(
            output=out,
            iou_threshold=iou_threshold_optimization,
            score_threshold=score_threshold_optimization
        )

        y_pred = evaluation_preds

        # attach optimized patch to the label target boxes
        target_boxes = [
            pred["boxes"][pred["labels"] == attack.config.label_target]
            for pred in optimization_preds
        ]

        with torch.no_grad():
            x_adv = torch.stack([
                attack.apply_patch(batch[i], target_boxes[i])
                for i in range(batch.shape[0])
            ])
        x_adv = torch.clamp(x_adv, min=attack.config.lower_bound, max=attack.config.upper_bound)

        with torch.no_grad():
            out_adv = model(x_adv)

        y_pred_adv = detection_predictions(
            output=out_adv,
            iou_threshold=iou_threshold_evaluation,
            score_threshold=score_threshold_evaluation
        )

        _log_batch(
            logger,
            original_input=batch,
            adversarial_input=x_adv,
            original_prediction=y_pred,
            adversarial_prediction=y_pred_adv,
        )

        statistics.update(
            x=batch.detach(),
            x_adv=x_adv.detach(),
            y=label,
            y_pred=y_pred,
            y_pred_adv=y_pred_adv,
            y_target=y_pred,
            out=out,
            out_adv=out_adv,
        )

    metric_states: dict[str, dict[str, Any]] = statistics.get_raw_state()
    statistics.update_aggregate(metric_states)

    return statistics
