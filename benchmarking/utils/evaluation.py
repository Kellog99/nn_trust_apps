from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import itertools
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from benchmarking.utils.attack import _create_atk
from models import JobResult
from models.reports import ParameterLog
from nn_trust import ModelAdapter, StatisticComposer
from nn_trust.attack import EvasionAttack
from nn_trust.utils import PyTorchCheckpointLogger

from nn_trust.attack.detection_utils import nms


def evaluate_attack(
        dataloader: DataLoader,
        model: ModelAdapter,
        attack: EvasionAttack | dict,
        statistics: StatisticComposer,
        device: torch.device,
        verbose: bool = False,
        output_path: Optional[str | Path] = None,
        max_saved_elements: int = 10,
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
            max_saved_elements: Optional maximum number of elements to save for each variable.
                Pass an integer for the same limit on every variable, or a dict keyed by variable name.
                The default preserves the current behavior of saving one element. Pass ``None`` to save all.
            output_path:
    """
    task = model.task

    ### PREPARE EXECUTION ###
    if isinstance(attack, dict):
        attack = _create_atk(
            attack=attack,
            model=model,
            out_path=output_path,
            device=device
        )

    if output_path is None:
        output_path = Path("./tmp") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if isinstance(output_path, str):
        output_path = Path(output_path)

    output_path = output_path.expanduser().resolve() / attack.__class__.__name__.lower().removesuffix("attack")

    logger: PyTorchCheckpointLogger = PyTorchCheckpointLogger(
        path=output_path,
        max_artifact={
            "original_input": max_saved_elements if max_saved_elements else 1,
            "adversarial_input": max_saved_elements if max_saved_elements else 1
        }
    )

    atk_id: str = attack.__class__.__name__.lower().removesuffix("attack")

    # --- FIX 1: peek the first batch without discarding it ---
    # `next(iter(dataloader))` used to build a *second*, independent iterator;
    # for IterableDataset-backed loaders this can drain the source (or, more
    # subtly, desynchronize state) so the loop below iterates zero times.
    base_iter = iter(dataloader)
    first_batch, first_label = next(base_iter)
    full_iter = itertools.chain([(first_batch, first_label)], base_iter)

    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
    if verbose:
        progress_bar = enumerate(
            tqdm(full_iter, total=total_batches, desc=f"Attack {repr(attack)} for model {model.name}")
        )
    else:
        progress_bar = enumerate(full_iter)

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

                x_adv = attack.generate(x=batch, y=out).detach()

                for b in range(batch.shape[0]):
                    logger.log(tag="original_input", data=batch[b])
                    logger.log(tag="adversarial_input", data=x_adv[b])


                boxes, scores = out
                num_classes = scores.shape[-1]

                iou_threshold = attack.config.iou_threshold_targeted
                score_threshold = attack.config.score_threshold_targeted
                targeted = attack.config.targeted
                label_target = attack.config.label_target


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

    logger.close()
    attack.logger.close()

    result = statistics.compute()
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
            for key, value in attack.config.__class__.model_fields.items()
            if key != "model" and isinstance(atk_parameters.get(key, None), (int, float, bool))
        ],
    )

    return job_result
