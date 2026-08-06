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
from nn_trust.target import AvoidOnehotTarget


def _create_atk(
        attack: dict,
        model: ModelAdapter,
        device: torch.device,
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
        task=Task.Classification,
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

    ### PREPARE EXECUTION ###
    if isinstance(attack, dict):
        attack = _create_atk(
            attack=attack,
            model=model,
            device=device
        )

    atk_id: str = attack.__class__.__name__.lower().removesuffix("attack")

    batch, _ = next(iter(dataloader))
    num_classes: int = model(batch.to(device)).shape[-1]
    if verbose:
        progress_bar = enumerate(tqdm(dataloader, desc=f"Attack {repr(attack)} for model {model.name}"))
    else:
        progress_bar = enumerate(dataloader)

    for idx, (batch, label) in progress_bar:

        batch = batch.to(device)
        label = label.to(device)

        target = AvoidOnehotTarget(num_classes=num_classes)(label.tolist()).to(batch.device)

        ############## Generating the adversarial image ##############
        x_adv = attack.generate(
            x=batch,
            y=target
        ).detach()
        ##############################################################

        with torch.no_grad():
            out = model(batch)
            out_adv = model(x_adv)

        y_pred_adv = out_adv.argmax(dim=-1)
        y_pred = out.argmax(dim=-1)

        # adapt metrics counting for reference or standard attack
        correct_mask = torch.eq(label, y_pred)
        if atk_id == "reference":
            y_pred = label

        elif torch.any(correct_mask):
            if not torch.all(correct_mask):
                label = label[correct_mask]
                x_adv = x_adv[correct_mask]
                batch = batch[correct_mask]
                out = out[correct_mask]
                out_adv = out_adv[correct_mask]
                y_pred = y_pred[correct_mask]
                y_pred_adv = y_pred_adv[correct_mask]

        else:
            continue  # skip iteration, no statistic update for this batch

        input_stat = {
            'x_adv': x_adv.detach(),
            'x': batch.detach(),
            'y': label,
            'y_target': label,
            'y_pred': y_pred,
            'y_pred_adv': y_pred_adv
        }
        statistics.update(**input_stat)

    result = statistics.compute()
    # In case the identity attack is performed, then no global statistics is used
    if atk_id != 'identitybaseline':
        metric_states: dict[str, dict[str, Any]] = statistics.get_raw_state()
        statistics.update_aggregate(metric_states)
    statistics.reset()
    atk_parameters: dict = attack.config.model_dump()
    out = JobResult(
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
    return out


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
