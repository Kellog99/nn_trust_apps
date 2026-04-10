from enum import Enum
from typing import Optional, Callable, Literal
import pathlib
import importlib

import torch
import torch.nn as nn



class AttackException(Exception):
    pass


class Knowledge(Enum):
    White: int = 0
    Black: int = 1

    @classmethod
    def from_str(cls, knowledge_str):
        match knowledge_str.lower():
            case "white":
                return cls.White
            case "black":
                return cls.Black
            case _:
                raise ValueError(f"Unknown knowledge: {knowledge_str}.")


class AttackType(Enum):
    Physical: int = 0
    Digital: int = 1

    @classmethod
    def from_str(cls, attack_type_str):
        match attack_type_str.lower():
            case "physical":
                return cls.Physical
            case "digital":
                return cls.Digital
            case _:
                raise ValueError(f"Unknown attack type: {attack_type_str}.")


class Task(Enum):
    Classification = 0
    Segmentation = 1
    Detection = 2

    @classmethod
    def from_str(cls, task_str):
        match task_str.lower():
            case "classification":
                return cls.Classification
            case "segmentation":
                return cls.Segmentation
            case "detection":
                return cls.Detection
            case _:
                raise ValueError(f"Unknown task: {task_str}.")


class ModelAdapter(nn.Module):
    """
    ModelAdapter provides the possibility to use the library with any kind of model even if those does not match the
    expected inputs and outputs by this library.
    It can be extended to adapt the model.
    """

    def __init__(
            self,
            model: Optional[nn.Module],
            name: Optional[str] = None,
            threat_model: Knowledge | str = Knowledge.White,
            task: Task | str = Task.Classification,
            transform: Optional[Callable] = None,
            **kwargs,
    ):
        super().__init__()
        self.model = model
        self.threat_model = Knowledge.from_str(threat_model) if isinstance(threat_model, str) else threat_model
        # NOTE: remove if default_cfg is not used in benchmarking.py. For now I leave it.
        if model:
            self.default_cfg = model.default_cfg if "default_cfg" in vars(model) else None
        try:
            self.name = name if name is not None else model._get_name() if model is not None else "name"
        except AttributeError:
            self.name = "name"
        self.transform = transform
        self.task = Task.from_str(task) if isinstance(task, str) else task

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""The `ModelAdapter` calls the backbone model with further restrictions
        on gradient computation depending on the specified ``Knowledge``.
        """
        if self.transform:
            x = self.transform(x.clamp(-1.0, 1.0))
        if self.threat_model == Knowledge.Black:
            with torch.no_grad():
                return self.model(x)
        else:
            return self.model(x)


