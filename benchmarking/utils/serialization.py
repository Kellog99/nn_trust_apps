import json
from pathlib import Path

import torch


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
