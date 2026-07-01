"""App-owned privacy model registry.

Registry and builder utilities for privacy benchmarking models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn

from nn_trust import Task


class FaceMLP(nn.Module):
    """Small MLP for embedding/vector privacy datasets."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        return self.fc2(torch.relu(self.fc1(x)))


class PropertyInferenceMLP(nn.Module):
    """Fully connected target/shadow model for CelebA embedding property inference."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def _build_face_mlp(*, num_classes: int, input_dim: int = 512, hidden_dim: int = 3000, **_: Any) -> nn.Module:
    return FaceMLP(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)


def _build_property_mlp(*, num_classes: int, input_dim: int = 512, **_: Any) -> nn.Module:
    return PropertyInferenceMLP(input_dim=input_dim, num_classes=num_classes)


def _build_resnet18(*, num_classes: int, **_: Any) -> nn.Module:
    import torchvision

    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@dataclass
class PrivacyModelSpec:
    model_id: str
    builder: Callable[..., nn.Module]
    name: str | None = None
    description: str | None = None
    task: set[Task] = field(default_factory=lambda: {Task.Classification})
    metadata: dict[str, Any] = field(default_factory=dict)

    def info(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "name": self.name or self.model_id,
            "description": self.description or f"Privacy model '{self.model_id}'",
            "task": self.task,
            **self.metadata,
        }


class AppPrivacyModelFactory:
    """Concrete model factory owned by nn_trust_apps."""

    def __init__(self) -> None:
        self._registry: dict[str, PrivacyModelSpec] = {}

    def register(self, spec: PrivacyModelSpec) -> None:
        self._registry[spec.model_id] = spec

    def list_specs(self) -> list[PrivacyModelSpec]:
        return list(self._registry.values())

    def build_model(self, *, model_id: str, num_classes: int, **kwargs: Any) -> nn.Module:
        spec = self._registry.get(model_id)
        if spec is None:
            raise ValueError(f"Unknown privacy model '{model_id}'. Registered: {sorted(self._registry)}")
        return spec.builder(num_classes=num_classes, **kwargs)

    def get_model_info(self, model_id: str) -> dict[str, Any]:
        spec = self._registry.get(model_id)
        if spec is None:
            raise ValueError(f"Unknown privacy model '{model_id}'. Registered: {sorted(self._registry)}")
        return spec.info()


def build_app_privacy_model_factory() -> AppPrivacyModelFactory:
    factory = AppPrivacyModelFactory()
    factory.register(
        PrivacyModelSpec(
            model_id="resnet18",
            builder=_build_resnet18,
            description="Torchvision ResNet-18 with task-specific classifier head.",
            metadata={"input_dim_from_data": False, "use_embeddings": False},
        )
    )
    factory.register(
        PrivacyModelSpec(
            model_id="property_mlp",
            builder=_build_property_mlp,
            description="512 -> 64 -> 16 -> task-head MLP for embedding property inference.",
            metadata={
                "input_dim": 512,
                "input_dim_from_data": True,
                "use_embeddings": True,
            },
        )
    )
    factory.register(
        PrivacyModelSpec(
            model_id="face_mlp",
            builder=_build_face_mlp,
            description="Two-layer MLP for flattened face embeddings or images.",
            metadata={
                "input_dim": 512,
                "hidden_dim": 3000,
                "input_dim_from_data": True,
                "use_embeddings": False,
            },
        )
    )
    return factory


__all__ = [
    "AppPrivacyModelFactory",
    "FaceMLP",
    "PrivacyModelSpec",
    "PropertyInferenceMLP",
    "build_app_privacy_model_factory",
]
