import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import itertools
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models import JobResult
from models.reports import ParameterLog
from nn_trust import ModelAdapter, AttackFactory as EAF, Task, EvasionAttack, StatisticComposer
from nn_trust.target import AvoidOnehotTarget
from nn_trust.utils import PyTorchCheckpointLogger


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

    job_result = JobResult(
        id=attack_id,
        status="pending",
        progress=0,
        total=len(dataloader.dataset)
    )
    with open(res_path, "w") as f:
        json.dump(job_result.model_dump(), f)

    ### PREPARE EXECUTION ###
    parameters: dict = parameters or {}
    attack: EvasionAttack = EAF.create(
        class_id=attack_id,
        model=model,
        device=device,
        task=Task.Classification,
        **parameters
    )

    logger: PyTorchCheckpointLogger = PyTorchCheckpointLogger(
        path=output_path,
        max_artifact={
            "original_input": max_saved_elements if max_saved_elements else 1,
            "adversarial_input": max_saved_elements if max_saved_elements else 1
        }
    )

    base_iter = iter(dataloader)
    try:
        first_batch, first_label = next(base_iter)
    except StopIteration as exc:
        raise ValueError("The dataloader produced no batches; cannot evaluate an attack.") from exc
    num_classes: int = model(first_batch.to(device)).shape[-1]

    pbar = tqdm(
        dataloader,
        desc=f"Attack {repr(attack)}",
    ) if verbose else dataloader

    job_result.status = "in progress"
    for batch, label in pbar:
        processed_count = batch.shape[0]
        batch = batch.to(device)
        label = label.to(device)

        target = AvoidOnehotTarget(num_classes=num_classes)(label.tolist()).to(batch.device)

        ############## Generating the adversarial image ##############
        x_adv = attack.generate(
            x=batch,
            y=target
        ).detach()

        for b in range(batch.shape[0]):
            logger.log(tag="original_input", data=batch[b])
            logger.log(tag="adversarial_input", data=x_adv[b])
        ##############################################################

        with torch.no_grad():
            model_output = model(batch)
            adversarial_output = model(x_adv)

            y_pred_adv = adversarial_output.argmax(dim=-1)
            y_pred = model_output.argmax(dim=-1)

        correct_mask = torch.eq(label, y_pred)
        if attack_id == "identitybaseline":
            y_pred = label
        elif torch.any(correct_mask):
            if not torch.all(correct_mask):
                label = label[correct_mask]
                x_adv = x_adv[correct_mask]
                batch = batch[correct_mask]
                model_output = model_output[correct_mask]
                adversarial_output = adversarial_output[correct_mask]
                y_pred = y_pred[correct_mask]
                y_pred_adv = y_pred_adv[correct_mask]

        input_stat = {
            'x_adv': x_adv.detach(),
            'x': batch.detach(),
            'y': label,
            'y_target': label,
            'y_pred': y_pred,
            'y_pred_adv': y_pred_adv
        }
        statistics.update(**input_stat)
        job_result.status = "in progress"
        job_result.progress += processed_count
        with open(res_path, "w") as f:
            json.dump(job_result.model_dump(), f)

    logger.close()
    attack.logger.close()

    result = statistics.compute()
    if attack_id != 'identitybaseline':
        metric_states: dict[str, dict[str, Any]] = statistics.get_raw_state()
        statistics.update_aggregate(metric_states)

    statistics.reset()
    atk_parameters: dict = attack.config.model_dump()
    job_result = JobResult(
        id=attack_id,
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
        progress=job_result.progress,
        total=job_result.total,
        status="finished"
    )

    with open(output_path / "job_results.json", "w") as f:
        json.dump(job_result.model_dump(), f)
    return job_result
