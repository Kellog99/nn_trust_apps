import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from nn_trust import Task
from nn_trust.utils import Logger, to_device
from benchmarking.utils.advyolo_evaluation import evaluate_frozen_advyolo, detection_predictions


def run_evaluation_pipeline(
    *, model, attack, attack_id, dataloader, statistics, device,
    logger=None, output_path=None, on_batch=None, verbose=False,
):
    """Execute one attack and update statistics.
    """
    task = model.task
    if task not in (Task.Classification, Task.Detection):
        raise NotImplementedError(f"{task} not supported yet.")
    logger = Logger() if logger is None else logger
    if attack_id == "advyoloevasion":
        if output_path is None:
            raise ValueError("An output_path is required for AdvYOLO evaluation.")
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        attack.generate(gen_train=dataloader)
        return evaluate_frozen_advyolo(
            dataloader=dataloader, model=model, attack=attack,
            statistics=statistics, logger=logger, device=device, output_path=output_path,
        )

    batches = tqdm(dataloader, desc=f"Attack {attack!r}") if verbose else dataloader
    for completed_iterations, (batch, label) in enumerate(batches, start=1):
        iteration_start = time.perf_counter()

        if not isinstance(batch, torch.Tensor):
            batch = torch.stack(batch)
        batch = batch.to(device)
        label = to_device(label, device)

        with torch.no_grad():
            out = model(batch)

        x_adv = attack.generate(x=batch, y=out).detach()
        with torch.no_grad():
            out_adv = model(x_adv)

        if task == Task.Classification:
            # The identity baseline measures accuracy against ground truth.
            y_pred = out.argmax(dim=-1)
            y_pred_adv = out_adv.argmax(dim=-1)
            y_target = y_pred
            if attack.config.targeted:
                y_target = ((y_pred + 1) % out.shape[-1])
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
        if on_batch is not None:
            on_batch(completed_iterations, len(batch), time.perf_counter() - iteration_start)

    return statistics
