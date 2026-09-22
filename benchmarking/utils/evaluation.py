import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

<<<<<<< HEAD
=======
from benchmarking.utils.attack import _create_atk
from benchmarking.utils.advyolo_evaluation import evaluate_frozen_advyolo
>>>>>>> dev/benchmarking
from models import JobResult
from models.reports import ParameterLog
from nn_trust import ModelAdapter, AttackFactory as EAF, Task, EvasionAttack, StatisticComposer
from nn_trust.target import AvoidOnehotTarget
from nn_trust.utils import PyTorchCheckpointLogger

<<<<<<< HEAD
=======
from nn_trust.attack.utils.detection import nms
>>>>>>> dev/benchmarking

def evaluate_attack(
        dataloader: DataLoader,
        model: ModelAdapter,
        statistics: StatisticComposer,
        attack_id: str,
        parameters: Optional[dict[str, Any]] = None,
        device: torch.device = torch.device("cpu"),
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
            attack_id: Attack to do on the (model, dataset)
            parameters: parameters of the attack
            statistics: The statistic composer that computes all the metrics that are required
            verbose
            device: device where the computation will be done
            max_saved_elements: Optional maximum number of elements to save for each variable.
                Pass an integer for the same limit on every variable, or a dict keyed by variable name.
                The default preserves the current behavior of saving one element. Pass ``None`` to save all.
            output_path:
    """
    ### Creating the dumping file ###
    if output_path is None:
        output_path = Path("./tmp") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if isinstance(output_path, str):
        output_path = Path(output_path)
    output_path: Path = output_path.expanduser().resolve() / attack_id
    output_path.mkdir(parents=True, exist_ok=True)
    res_path: Path = output_path / "job_results.json"

    total = len(dataloader.dataset)
    job_result = JobResult(
        id=attack_id,
        status="pending",
        progress=0,
        total=total
    )
    job_result.save(res_path)

    parameters = parameters or {}
    execution_start = time.perf_counter()
    total_iterations = len(dataloader)

<<<<<<< HEAD
    attack: Optional[EvasionAttack] = None
    logger: Optional[PyTorchCheckpointLogger] = None

    try:
        job_result = job_result.model_copy(update={"status": "in progress"})
        job_result.save(res_path)

        ### PREPARE EXECUTION ###
        attack = EAF.create(
            class_id=attack_id,
            model=model,
            device=device,
            task=Task.Classification,
            **parameters
        )

        logger = PyTorchCheckpointLogger(
            path=output_path,
            max_artifact={
                "original_input": max_saved_elements if max_saved_elements else 1,
                "adversarial_input": max_saved_elements if max_saved_elements else 1
            }
        )

        pbar = tqdm(dataloader, desc=f"Attack {repr(attack)}") if verbose else dataloader

        num_classes: Optional[int] = None

        for completed_iterations, (batch, label) in enumerate(pbar, start=1):
            iteration_start = time.perf_counter()
            processed_count = batch.shape[0]
            batch = batch.to(device)
            label = label.to(device)

            with torch.no_grad():
                model_output = model(batch)
            if num_classes is None:
                num_classes: int = model_output.shape[-1]

            target = AvoidOnehotTarget(num_classes=num_classes)(label.tolist()).to(batch.device)

            ############## Generating the adversarial image ##############
            x_adv = attack.generate(x=batch, y=target).detach()
=======
    # --- FIX 1: peek the first batch without discarding it ---
    # `next(iter(dataloader))` used to build a *second*, independent iterator;
    # for IterableDataset-backed loaders this can drain the source (or, more
    # subtly, desynchronize state) so the loop below iterates zero times.
    if atk_id != "advyoloevasion":
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

                    boxes, scores = out

                    post_nms_preds = nms(
                        {
                        "boxes": boxes,
                        "scores": scores.max(dim=-1).values,
                        "cls_scores": scores,
                        },
                    iou_threshold=attack.config.iou_threshold_evaluation, # compare one reference predicted bounding box with the other predicted bounding boxes. If the IoU between the two boxes is over the threshold, discards the box with the lower score. Ones all the remaining boxes are compared, we select the next reference box. As such, increasing the threshold increases the nubmer of final predicted bounding boxes by the model, because less boxes are discarded
                    score_threshold=attack.config.score_threshold_evaluation, # filter out all the predicted bounding boxes whose score is below the threshold. As such, increaidng the threshold increases the number of final predicted bounding boxes, because less boxes are discarded
                    )

                    # convert post_nms_preds to targets for metrics
                    y_pred = [
                    {
                        "boxes": pred["boxes"],
                        "labels": pred["labels"],
                        "scores": pred["scores"],
                        "cls_scores": pred["cls_scores"],
                    }
                    for pred in post_nms_preds    
                    ]

                    x_adv = attack.generate(x=batch, y=out).detach()

                    with torch.no_grad():
                        out_adv = model(x_adv)
                    
                    # convert post_nms_preds to the desired targets for metrics
                    targeted = attack.config.targeted
                    if targeted == True:
                        num_classes = scores.shape[-1]
                        label_target = attack.config.label_target
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
                        iou_threshold=attack.config.iou_threshold_evaluation,
                        score_threshold=attack.config.score_threshold_evaluation,
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
>>>>>>> dev/benchmarking

            for b in range(batch.shape[0]):
                logger.log(tag="original_input", data=batch[b])
                logger.log(tag="adversarial_input", data=x_adv[b])
            ##############################################################

            with torch.no_grad():
                adversarial_output = model(x_adv)
                y_pred_adv = adversarial_output.argmax(dim=-1)
                y_pred = model_output.argmax(dim=-1)

<<<<<<< HEAD
            if attack_id == "identitybaseline":
                y_pred = label
        
            statistics.update(
                x_adv=x_adv.detach(),
                x=batch.detach(),
                y=label,
                y_target=label,
                y_pred=y_pred,
                y_pred_adv=y_pred_adv,
            )

            iteration_time = time.perf_counter() - iteration_start
            execution_time = time.perf_counter() - execution_start
            job_result = job_result.model_copy(update={
                "status": "in progress",
                "progress": job_result.progress + processed_count,
                "iteration_time": iteration_time,
                "execution_time": execution_time,
                "estimated_execution_time": execution_time / completed_iterations * total_iterations,
            })
            job_result.save(res_path)

    except Exception as exc:
        job_result = job_result.model_copy(
            update={
                "status": "error",
                "error": str(exc)
            }
        )
        job_result.save(res_path)
        raise
    finally:
        if logger is not None:
            logger.close()
        if attack is not None:
            attack.logger.close()

    result = statistics.compute()
    if attack_id != 'identitybaseline':
        metric_states: dict[str, dict[str, Any]] = statistics.get_raw_state()
        statistics.update_aggregate(metric_states)
    statistics.reset()

=======
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

        if atk_id != 'identitybaseline':
            metric_states: dict[str, dict[str, Any]] = statistics.get_raw_state()
            statistics.update_aggregate(metric_states)


    else:

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

    result = statistics.compute()
    statistics.reset()
            
    logger.close()
    attack.logger.close()

>>>>>>> dev/benchmarking
    atk_parameters: dict = attack.config.model_dump()
    job_result = job_result.model_copy(update={
        "result": result,
        "parameters": [
            ParameterLog(
                id=key,
                name=value.title,
                description=value.description,
                value=atk_parameters[key]
            )
            for key, value in attack.config.__class__.model_fields.items()
            if key != "model" and isinstance(atk_parameters.get(key, None), (int, float, bool))
        ],
        "execution_time": time.perf_counter() - execution_start,
        "status": "finished",
    })
    job_result.save(res_path)
    return job_result
