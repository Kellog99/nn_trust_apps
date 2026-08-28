from datetime import datetime
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models import JobResult
from models.reports import ParameterLog
from nn_trust import ModelAdapter, StatisticComposer, LossComposer, AttackFactory as EAF, Task
from nn_trust.attack import EvasionAttack
from nn_trust.target import AvoidOnehotTarget
from nn_trust.utils import PyTorchCheckpointLogger


def _create_atk(
        attack: dict,
        model: ModelAdapter,
        device: torch.device,
        out_path: Optional[str | Path] = None,
) -> EvasionAttack:
    atk_id: str | None = attack.get("id", None) or attack.get("name", None)
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
            "original_input": max_saved_elements,
            "adversarial_input": max_saved_elements
        }
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

        # Logging all the various variables
        for b in range(batch.shape[0]):
            logger.log(
                tag="original_input",
                data=batch[b],
            )
            logger.log(
                tag="adversarial_input",
                data=x_adv[b],
            )
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

    # Once the attack is terminated, it is possible to close the Logger.
    logger.close()
    attack.logger.close()

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
            for key, value in attack.config.__class__.model_fields.items()
            if key != "model" and isinstance(atk_parameters.get(key, None), (int, float, bool))
        ],
    )
    return out
