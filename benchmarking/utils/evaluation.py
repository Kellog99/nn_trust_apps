import json
import os
import pathlib
import pickle
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from models import JobResult, ParametersProps
from models.reports import ParameterLog
from nn_trust import ModelAdapter, StatisticComposer, LossComposer, AttackFactory as EAF, Task
from nn_trust.attack import EvasionAttack
from nn_trust.attack.detection_utils import nms


def _create_atk(
        attack: dict,
        model: ModelAdapter,
        device: torch.device,
        task: Task
) -> EvasionAttack:
    atk_id: str = attack.get("id", None) or attack.get("name", None)
    if atk_id is None:
        raise ValueError("No id for instantiate the attack")

    atk_config = {
        "name": attack.get("id", None),
        "id": attack.get("id", None),
        **{
            key: value
            for key, value in attack.items()  # This is for extracting all the eventual parameters that are passed
            if key != "id"
        },
    }
    # Checking whether some losses have to be set
    if atk_config.get("losses", None) is not None:
        # If losses are specified, convert them to Loss objects
        atk_config['loss'] = LossComposer(
            losses=atk_config['losses'],
            weights=atk_config.get('loss_weights', [1.0] * len(atk_config['losses'])),
        )

    return EAF.create(
        class_id=atk_id,
        model=model,
        device=device,
        task=model.task,
        **atk_config
    )


def evaluate_attack(
        dataloader: torch.utils.data.DataLoader,
        model: ModelAdapter,
        attack: EvasionAttack | dict,
        statistics: StatisticComposer,
        device: torch.device,
        verbose: bool = False,
) -> JobResult:
    """
    Evaluate the model's vulnerability on the attack that is passed.
    Since this is for performance purpose, it is assumed that it is not targeted.
    Moreover, it updates the global statistics.

        Args:
            dataloader: dataset to use
            model: target model
            attack: Attack to do on the (model, dataset)
            statistics: The statistic composer that computes all the metrics that are required
            verbose
            device: device where the computation will be done
    """
    task = model.task

    ### PREPARE EXECUTION ###
    if isinstance(attack, dict):
        attack = _create_atk(
            attack=attack,
            model=model,
            device=device,
            task=task
        )

    atk_id: str = attack.__class__.__name__.lower().removesuffix("attack")

    #batch, _ = next(iter(dataloader))
    #num_classes: int = model(batch.to(device)).shape[-1]
    if verbose:
        progress_bar = enumerate(tqdm(dataloader, desc=f"Attack {repr(attack)} for model {model.name}"))
    else:
        progress_bar = enumerate(dataloader)

    for idx, (batch, label) in progress_bar:

        y_target = None
        match task:
            case Task.Detection:
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
                num_classes = scores.shape[-1]

                iou_threshold = attack.config.iou_threshold_targeted
                score_threshold = attack.config.score_threshold_targeted
                targeted = attack.config.targeted
                label_target = attack.config.label_target

                x_adv = attack.generate(x=batch, y=out)

                with torch.no_grad():
                    out_adv = model(x_adv)

                # apply nms on predictions from original images
                # We give a high iou_threshold and low score_threshold so that we give as many scores as possible to coreectly compute map
                boxes, scores = out
                post_nms_preds = nms(
                    {
                    "boxes": boxes,
                    "scores": scores.max(dim=-1).values,
                    "cls_scores": scores,
                    },
                iou_threshold=iou_threshold, # compare one reference predicted bounding box with the other predicted bounding boxes. If the IoU between the two boxes is over the threshold, discards the box with the lower score. Ones all the remaining boxes are compared, we select the next reference box. As such, increasing the threshold increases the nubmer of final predicted bounding boxes by the model, because less boxes are discarded
                score_threshold=score_threshold, # filter out all the predicted bounding boxes whose score is below the threshold. As such, increaidng the threshold increases the number of final predicted bounding boxes, because less boxes are discarded
                )

                # convert post_nms_preds to targets for metrics
                y_pred = [
                {
                    "boxes": pred["boxes"],
                    "labels": pred["labels"],
                }
                for pred in post_nms_preds    
                ]
                
                # convert post_nms_preds to the desired targets for metrics
                if targeted == True:
                    target_class = (label_target + 1) % num_classes

                    y_target = [
                    {
                        "boxes": pred["boxes"],
                        "labels": torch.where(
                            pred["labels"] == label_target, # the condition to satisfy
                            torch.full_like(pred["labels"], target_class), # insert the target_class where the condition is satisfied
                            pred["labels"], # insert the original pred["labels"] values where the condition is not satisfied
                        ),
                    }
                    for pred in post_nms_preds
                    ]
                else:
                    y_target = y_pred

                # apply nms on predictions from adversarial images
                # We give a high iou_threshold and low score_threshold so that we give as many scores as possible to coreectly compute map
                adv_boxes, adv_scores = out_adv
                y_pred_adv = nms(
                    {
                        "boxes": adv_boxes,
                        "scores": adv_scores.max(dim=-1).values,
                        "cls_scores": adv_scores,
                    },
                    iou_threshold=iou_threshold,
                    score_threshold=score_threshold,
                )

            case Task.Classification:
                batch = batch.to(device)
                label = label.to(device)

                with torch.no_grad():
                    out = model(batch)

                y_pred = out.argmax(dim=-1)
                correct_mask = torch.eq(label, y_pred)

                if atk_id != "identitybaseline":
                    if not torch.any(correct_mask):
                        continue

                    batch = batch[correct_mask]
                    label = label[correct_mask]
                    out = out[correct_mask]
                    y_pred = y_pred[correct_mask]

                x_adv = attack.generate(x=batch, y=out.detach()).detach()

                with torch.no_grad():
                    out_adv = model(x_adv)

                y_pred_adv = out_adv.argmax(dim=-1)

            case _:
                raise NotImplementedError(f"{task} not supported yet.")


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

    result = statistics.compute()
    # In case the identity attack is performed, then no global statistics is used
    if atk_id != 'identitybaseline':
        metric_states: dict[str, dict[str, Any]] = statistics.get_raw_state()
        statistics.update_aggregate(metric_states)
    statistics.reset()
    atk_parameters: dict = attack.config.model_dump()
    job_result = JobResult(
        id=atk_id,
        result=result,
        parameters=[
            ParameterLog(
                id=key,
                name=value.title,
                description=value.description,
                value=atk_parameters[key]
            )
            for key, value in attack.config.model_fields.items()
            if key != "model" and isinstance(atk_parameters.get(key, None), (int, float, bool))
        ],
    )
    return job_result


def make_json_serializable(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()

    if isinstance(value, torch.Size):
        return list(value)

    if isinstance(value, torch.device):
        return str(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [make_json_serializable(item) for item in value]

    if isinstance(value, tuple):
        return [make_json_serializable(item) for item in value]

    # if json can already save it, keep it. If not, convert it to a string
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)
