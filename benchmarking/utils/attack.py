from pathlib import Path
from typing import Optional

import torch

from nn_trust import ModelAdapter, LossComposer, AttackFactory as EAF, Task
from nn_trust.attack import EvasionAttack


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
