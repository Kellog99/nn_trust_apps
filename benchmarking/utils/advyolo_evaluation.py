from typing import Any

import torch
import torchvision

from nn_trust.attack.utils.detection import nms


def evaluate_frozen_advyolo(
    dataloader,
    model,
    attack,
    statistics,
    logger,
    device,
    output_path,
):
    patch = attack.patch.detach().cpu().clone()

    if attack.config.save == True:
        torch.save(
            {"patch": patch},
            output_path / "patch.pt",
        )
        torchvision.utils.save_image(
            patch.clamp(0, 1),
            output_path / "patch.png",
        )

    # Evaluate the final frozen patch
    for batch, label in dataloader:
        batch = torch.stack(batch).to(device)

        label = [
            {
                "boxes": label_["boxes"].to(device),
                "labels": label_["labels"].to(device),
            }
            for label_ in label
        ]

        with torch.no_grad():
            out = model(batch)

        boxes, scores = out

        output_dict = {
            "boxes": boxes,
            "scores": scores.max(dim=-1).values,
            "cls_scores": scores,
            }
        
        evaluation_preds = nms(
            output_dict,
            iou_threshold=attack.config.iou_threshold_evaluation,
            score_threshold=attack.config.score_threshold_evaluation,
        )

        optimization_preds = nms(
            output_dict,
            iou_threshold=attack.config.iou_threshold_optimization,
            score_threshold=attack.config.score_threshold_optimization,
        )

        # convert post_nms_preds to targets for metrics
        y_pred = [
        {
            "boxes": pred["boxes"],
            "labels": pred["labels"],
            "scores": pred["scores"],
            "cls_scores": pred["cls_scores"],
        }
        for pred in evaluation_preds    
        ]

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

        x_adv = torch.clamp(
            x_adv,
            min=attack.config.lower_bound,
            max=attack.config.upper_bound,
        )

        with torch.no_grad():
            out_adv = model(x_adv)

        y_target = y_pred

        adv_boxes, adv_scores = out_adv
        y_pred_adv = nms(
            {
                "boxes": adv_boxes,
                "scores": adv_scores.max(dim=-1).values,
                "cls_scores": adv_scores,
            },
            iou_threshold=attack.config.iou_threshold_evaluation,
            score_threshold=attack.config.score_threshold_evaluation,
        )


        for b in range(batch.shape[0]):
            logger.log(tag="original_input", data=batch[b])
            logger.log(tag="adversarial_input", data=x_adv[b])
            logger.log(tag="original_prediction", data=y_pred[b])
            logger.log(tag="adversarial_prediction", data=y_pred_adv[b])

        input_stat = {
        'x_adv': x_adv.detach(),
        'x': batch.detach(),
        'y': label,
        'y_pred': y_pred,
        'y_pred_adv': y_pred_adv,
        'y_target': y_target,
        'out': out,
        'out_adv': out_adv
        }
        statistics.update(**input_stat)

    metric_states: dict[str, dict[str, Any]] = statistics.get_raw_state()
    statistics.update_aggregate(metric_states)

    result = statistics.compute()
    statistics.reset()

    return result

    