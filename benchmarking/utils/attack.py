from pathlib import Path
from typing import Any, Optional

import torch

from nn_trust import ModelAdapter, LossComposer, AttackFactory as EAF, Task
from nn_trust.attack import EvasionAttack


# ``RegisteredObject`` is the representation used by the API catalogue.  Its
# descriptive fields must not be passed to an attack constructor.  In
# particular, forwarding ``task`` clashes with the explicit factory argument
# below and prevents every job from starting.
_ATTACK_METADATA_FIELDS = {
    "id",
    "name",
    "description",
    "parameters",
    "task",
    "knowledge",
    "objective",
    "privacy_type",
    "actions",
}


def _attack_parameters(attack: dict[str, Any]) -> dict[str, Any]:
    """Return constructor arguments from either an API object or a compact spec."""
    parameters: dict[str, Any] = {}

    # API catalogue objects carry defaults nested in ``parameters``.  Explicit
    # top-level values below take precedence, so callers can override them.
    for parameter in attack.get("parameters", []):
        if isinstance(parameter, dict) and parameter.get("id") is not None and "default" in parameter:
            parameters[parameter["id"]] = parameter["default"]

    parameters.update({
        key: value
        for key, value in attack.items()
        if key not in _ATTACK_METADATA_FIELDS
    })
    return parameters


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
        "name": atk_id,
        "id": atk_id,
        **_attack_parameters(attack),
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
