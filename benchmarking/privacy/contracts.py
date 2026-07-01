"""App-side protocol contracts for privacy datasets, models, and training."""

from pathlib import Path
from typing import Protocol, Optional

import torch
from torch.utils.data import Dataset, Subset

from nn_trust import Task
from .dataset_registry import AppPrivacyDatasetFactory, build_app_privacy_dataset_factory
from .model_registry import build_app_privacy_model_factory

_default_dataset_factory: AppPrivacyDatasetFactory | None = None


# ---------------------------------------------------------------------------
# Dataset protocol interfaces
# ---------------------------------------------------------------------------

class PrivacyDatasetHandle(Protocol):
    """Minimal public dataset interface needed by app-side privacy protocols."""

    num_classes: int
    full_dataset: Dataset

    def build_subset(self, indices: list[int]) -> Subset:
        """Materialize a subset from caller-owned split indices."""


class PrivacyAttributeDatasetHandle(PrivacyDatasetHandle, Protocol):
    """Public extension for privacy datasets exposing binary attributes."""

    def get_binary_attribute_values(
            self,
            attribute_name: str,
            *,
            indices: Optional[list[int] | tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Return binary attribute values for the requested dataset indices."""


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def register_privacy_resources() -> None:
    """Populate the privacy model factory registries.

    Apps should call this before using model-building helpers.
    """
    global _default_dataset_factory
    #set_default_model_factory(build_app_privacy_model_factory())
    _default_dataset_factory = build_app_privacy_dataset_factory()


def get_privacy_dataset_factory() -> AppPrivacyDatasetFactory:
    """Return the configured app-side privacy dataset factory."""
    if _default_dataset_factory is None:
        raise RuntimeError("Privacy dataset factory is not configured. Call register_privacy_resources() first.")
    return _default_dataset_factory


def get_privacy_binary_attribute_values(
        dataset: PrivacyDatasetHandle,
        *,
        attribute_name: str,
        indices: list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Return one binary attribute column through the public privacy dataset API."""
    getter = getattr(dataset, "get_binary_attribute_values", None)
    if not callable(getter):
        raise TypeError(
            f"Privacy dataset '{type(dataset).__name__}' does not expose public binary attribute access."
        )

    normalized_indices = None if indices is None else list(int(index) for index in indices)
    values = getter(attribute_name, indices=normalized_indices)
    if not isinstance(values, torch.Tensor):
        raise TypeError(
            "Privacy dataset binary attribute access must return a torch.Tensor, "
            f"got {type(values).__name__}."
        )
    return values.long()


def resolve_privacy_model_task(model_id: str) -> Task:
    info = {}#get_default_model_factory().get_model_info(model_id)
    task_value = info.get("task")
    if not isinstance(task_value, set) or not all(isinstance(t, Task) for t in task_value):
        raise ValueError(
            f"Privacy target model '{model_id}' must expose a task set of Task values, got {task_value!r}.")
    tasks = frozenset(task_value)
    if len(tasks) != 1:
        raise ValueError(
            f"Privacy target model '{model_id}' must expose exactly one task, "
            f"got {sorted(t.name for t in tasks)}."
        )
    return next(iter(tasks))


def load_privacy_dataset(
        *,
        dataset_id: str,
        root: Path,
        seed: int = 42,
        task_attr: str | None = None,
        use_embeddings: bool = True,
        max_samples: int | None = None,
        **kwargs,
) -> PrivacyDatasetHandle:
    """Load one registered privacy dataset wrapper.

    Delegates to the app-side dataset factory, which must be injected
    before this function is called.
    """
    return get_privacy_dataset_factory().load_dataset(
        dataset_id=dataset_id,
        root=root,
        seed=seed,
        task_attr=task_attr,
        use_embeddings=use_embeddings,
        max_samples=max_samples,
        **kwargs,
    )
