"""App-side privacy target-model training and provenance resolution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from nn_trust.attack.utils.model_building import build_privacy_model
from nn_trust.attack.utils.training import (
    TrainerConfig,
    build_trainer,
)

from .contracts import PrivacyDatasetHandle
from .persistence import _ensure_dir, _write_json
from .job_models import PrivacyJobConfig, TargetModelSourceType
from .split_planning import MaterializedPrivacySplits

_TRAINED_MODEL_FILENAME = "best_model.pt"
_TRAINER_CONFIG_FILENAME = "trainer_config.json"
_TARGET_METADATA_FILENAME = "target_model_metadata.json"
_TARGET_TRAINING_IDENTITY_VERSION = 3


@dataclass
class LoadedPrivacyTargetModel:
    """Resolved target model plus provenance metadata for one privacy job."""

    model: nn.Module
    checkpoint_path: Path
    source_type: TargetModelSourceType
    training_recipe_id: str | None = None
    trained_now: bool = False


def _build_target_model_identity_payload(job: PrivacyJobConfig) -> dict[str, Any]:
    """Build the stable payload that defines one trained target model identity."""
    target_model_payload = job.target_model.model_dump(mode="json")
    target_model_payload.pop("checkpoint_path", None)
    target_model_payload.pop("shadow_model_id", None)
    return {
        "target_training_identity_version": _TARGET_TRAINING_IDENTITY_VERSION,
        "dataset": job.dataset.model_dump(mode="json"),
        "split_plan": job.split_plan.model_dump(mode="json"),
        "target_model": target_model_payload,
    }


def compute_privacy_target_model_fingerprint(job: PrivacyJobConfig) -> str:
    """Compute a stable fingerprint for one trained target model provenance."""
    identity_payload = _build_target_model_identity_payload(job)
    serialized_payload = json.dumps(identity_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()[:16]


def resolve_trained_target_output_dir(job: PrivacyJobConfig) -> Path:
    """Resolve the output directory for one trained privacy target model."""
    if job.target_model.source_type != TargetModelSourceType.TRAIN:
        raise ValueError("resolve_trained_target_output_dir only supports source_type='train'.")
    if job.target_model.checkpoint_path is not None:
        return Path(job.target_model.checkpoint_path).parent

    return (
        Path(job.dataset.root)
        / job.dataset.dataset_id
        / "models"
        / job.target_model.model_id
        / compute_privacy_target_model_fingerprint(job)
    )


def resolve_trained_target_checkpoint_path(job: PrivacyJobConfig) -> Path:
    """Resolve the checkpoint path for one trained privacy target model."""
    if job.target_model.source_type != TargetModelSourceType.TRAIN:
        raise ValueError("resolve_trained_target_checkpoint_path only supports source_type='train'.")
    if job.target_model.checkpoint_path is not None:
        return Path(job.target_model.checkpoint_path)
    return resolve_trained_target_output_dir(job) / _TRAINED_MODEL_FILENAME


def _resolve_target_training_seed(job: PrivacyJobConfig) -> int:
    overrides = job.target_model.training_overrides
    if overrides is not None and overrides.seed is not None:
        return int(overrides.seed)
    return int(job.split_plan.seed)


def resolve_target_trainer_config(
    job: PrivacyJobConfig,
    *,
    device: torch.device,
) -> TrainerConfig:
    """Resolve the trainer configuration for one train-backed privacy target model."""
    seed = _resolve_target_training_seed(job)
    overrides = job.target_model.training_overrides

    if job.target_model.training_recipe_id == "property_inference_shadow_match":
        from nn_trust.attack.privacy.property_inference import PropertyInferenceConfig
        defaults = PropertyInferenceConfig
        config = TrainerConfig(
            epochs=int(defaults.model_fields["shadow_epochs"].default),
            learning_rate=float(defaults.model_fields["shadow_lr"].default),
            batch_size=int(defaults.model_fields["shadow_batch_size"].default),
            seed=seed, verbose=bool(job.verbose), device=device,
        )
    else:
        config = TrainerConfig(seed=seed, verbose=bool(job.verbose), device=device)

    if overrides is None:
        return config
    update = overrides.to_update_mapping()
    update.setdefault("seed", seed)
    update.update(device=device, verbose=bool(job.verbose))
    return config.model_copy(update=update)


def _build_loader_from_dataset(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def _build_target_train_and_val_datasets(
    job: PrivacyJobConfig,
    *,
    dataset: PrivacyDatasetHandle,
    split_plan: MaterializedPrivacySplits,
) -> tuple[Dataset, Dataset]:
    train_indices = list(split_plan.target_train)
    if job.target_model.property_ratio is None:
        return dataset.build_subset(train_indices), dataset.build_subset(list(split_plan.target_val))

    # Property-filtered training: delegate to the dataset's own method
    target_pool_indices = [
        *split_plan.target_train,
        *split_plan.target_val,
        *split_plan.target_test,
    ]
    getter = getattr(dataset, "get_property_filtered_subset", None)
    if not callable(getter):
        raise TypeError(
            f"Privacy dataset '{type(dataset).__name__}' does not expose public property-filtered subset access."
        )
    train_subset = getter(
        property_attr=str(job.target_model.property_name),
        target_ratio=float(job.target_model.property_target_ratio),
        subset_size=len(train_indices),
        seed=_resolve_target_training_seed(job),
        indices=target_pool_indices,
    )
    selected_train_indices = set(getattr(train_subset, "indices", []))
    val_indices = [index for index in target_pool_indices if index not in selected_train_indices]
    val_size = len(split_plan.target_val)
    if len(val_indices) < val_size:
        raise RuntimeError(
            "Not enough target-pool samples remain to build a validation split disjoint from "
            "the property-filtered target training subset."
        )
    return train_subset, dataset.build_subset(val_indices[:val_size])


def train_privacy_target_model(
    job: PrivacyJobConfig,
    *,
    dataset: PrivacyDatasetHandle,
    split_plan: MaterializedPrivacySplits,
    device: torch.device,
    model_kwargs: dict[str, Any] | None = None,
) -> LoadedPrivacyTargetModel:
    """Train or load one target model owned by the app-side privacy executor."""
    if job.target_model.source_type != TargetModelSourceType.TRAIN:
        raise ValueError("train_privacy_target_model only supports source_type='train'.")

    checkpoint_path = resolve_trained_target_checkpoint_path(job)
    model_kwargs = model_kwargs or {}
    if checkpoint_path.exists():
        model = build_privacy_model(
            model_id=job.target_model.model_id,
            num_classes=dataset.num_classes,
            **model_kwargs,
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        model = model.to(device)
        model.eval()
        return LoadedPrivacyTargetModel(
            model=model,
            checkpoint_path=checkpoint_path,
            source_type=TargetModelSourceType.TRAIN,
            training_recipe_id=job.target_model.training_recipe_id,
            trained_now=False,
        )

    trainer_config = resolve_target_trainer_config(job, device=device)
    trainer = build_trainer(trainer_config)
    model = build_privacy_model(
        model_id=job.target_model.model_id,
        num_classes=dataset.num_classes,
        **model_kwargs,
    )

    train_dataset, _ = _build_target_train_and_val_datasets(
        job,
        dataset=dataset,
        split_plan=split_plan,
    )
    train_loader = _build_loader_from_dataset(
        train_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=True,
        device=device,
    )

    output_dir = _ensure_dir(resolve_trained_target_output_dir(job))
    model = trainer.fit(
        model=model,
        train_loader=train_loader,
    )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

    _write_json(
        output_dir / _TRAINER_CONFIG_FILENAME,
        trainer_config.model_dump(mode="json", exclude={"device"}),
        indent=2,
    )
    _write_json(
        output_dir / _TARGET_METADATA_FILENAME,
        {
            "model_id": job.target_model.model_id,
            "source_type": job.target_model.source_type.value,
            "training_recipe_id": job.target_model.training_recipe_id,
            "property_ratio": job.target_model.property_ratio,
            "property_name": job.target_model.property_name,
            "property_target_ratio": job.target_model.property_target_ratio,
            "trained_checkpoint_path": str(checkpoint_path),
            "dataset_id": job.dataset.dataset_id,
            "split_strategy": job.split_plan.strategy.value,
            "split_seed": int(job.split_plan.seed),
            "dataset_seed": int(job.dataset.seed),
            "target_split_sizes": {
                "train": len(split_plan.target_train),
                "val": len(split_plan.target_val),
                "test": len(split_plan.target_test),
            },
        },
        indent=2,
    )
    model = model.to(device)
    model.eval()
    return LoadedPrivacyTargetModel(
        model=model,
        checkpoint_path=checkpoint_path,
        source_type=TargetModelSourceType.TRAIN,
        training_recipe_id=job.target_model.training_recipe_id,
        trained_now=True,
    )


__all__ = [
    "LoadedPrivacyTargetModel",
    "compute_privacy_target_model_fingerprint",
    "resolve_target_trainer_config",
    "resolve_trained_target_checkpoint_path",
    "resolve_trained_target_output_dir",
    "train_privacy_target_model",
]
