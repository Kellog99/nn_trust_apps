"""Shared loading helpers for app-side privacy execution.

Contains DatasetWrapper and helpers to load privacy datasets, build
train/shadow/auxiliary splits, and construct PyTorch DataLoaders for
protcol execution.  Plugs into the nn_trust privacy benchmarking framework.
"""

from pathlib import Path
from typing import Any

import torch

from torch.utils.data import DataLoader

from nn_trust.attack import AttackFactory
from nn_trust.attack.privacy import register_privacy_attacks
from nn_trust.attack.utils.model_building import (
    build_privacy_model,
    get_default_model_factory,
    infer_data_derived_model_kwargs,
)

from .contracts import (
    PrivacyDatasetHandle,
    load_privacy_dataset,
    register_privacy_resources,
    resolve_privacy_model_task,
)
from .job_models import MaterializedPrivacySplits, PrivacyJobConfig, PrivacyProtocolId, PrivacySplitPlanConfig, PrivacySplitStrategy, TargetModelSourceType
from .target_training import LoadedPrivacyTargetModel, train_privacy_target_model


def ensure_privacy_registries() -> None:
    """Import privacy datasets, models, and attacks so factories are populated."""
    register_privacy_resources()
    register_privacy_attacks()


def load_privacy_dataset_wrapper(job: PrivacyJobConfig) -> PrivacyDatasetHandle:
    """Load one privacy dataset wrapper for app-side execution."""
    use_embeddings = job.dataset.use_embeddings
    if use_embeddings is None:
        info = get_default_model_factory().get_model_info(job.target_model.model_id)
        use_embeddings = bool(info.get("use_embeddings", not info.get("input_dim_from_data", False)))

    return load_privacy_dataset(
        dataset_id=job.dataset.dataset_id,
        root=job.dataset.root,
        seed=job.dataset.seed,
        task_attr=job.dataset.task_attr,
        use_embeddings=use_embeddings,
        max_samples=job.dataset.max_samples,
    )


def load_privacy_target_model(
    job: PrivacyJobConfig,
    *,
    dataset: PrivacyDatasetHandle,
    split_plan: MaterializedPrivacySplits,
    device: torch.device,
) -> LoadedPrivacyTargetModel:
    """Load one target model for a privacy job.

    Supports both checkpoint-backed and app-trained target-model provenance.
    """
    sample = dataset.full_dataset[0]
    sample_x = sample[0] if isinstance(sample, (tuple, list)) else sample
    if not isinstance(sample_x, torch.Tensor):
        sample_x = torch.as_tensor(sample_x)
    model_kwargs: dict[str, int] = infer_data_derived_model_kwargs(
        job.target_model.model_id,
        sample_x=sample_x,
    )

    if job.target_model.source_type == TargetModelSourceType.TRAIN:
        return train_privacy_target_model(
            job,
            dataset=dataset,
            split_plan=split_plan,
            device=device,
            model_kwargs=model_kwargs,
        )

    checkpoint_path = Path(job.target_model.checkpoint_path) if job.target_model.checkpoint_path else None
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Target checkpoint not found for model '{job.target_model.model_id}': {checkpoint_path}"
        )

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
        source_type=TargetModelSourceType.CHECKPOINT,
        training_recipe_id=None,
        trained_now=False,
    )


def resolve_shadow_model_id(job: PrivacyJobConfig) -> str:
    """Resolve the shadow-model architecture id for one privacy job."""
    return job.target_model.shadow_model_id or job.target_model.model_id


def resolve_surrogate_model_id(job: PrivacyJobConfig) -> str:
    """Resolve the surrogate-model architecture id for one extraction job."""
    configured_surrogate_model_id = job.attack.attack_params.get("surrogate_model_id")
    if configured_surrogate_model_id is not None:
        return str(configured_surrogate_model_id)
    return job.target_model.shadow_model_id or job.target_model.model_id


def build_privacy_loader(
    dataset: PrivacyDatasetHandle,
    indices: list[int],
    *,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    """Materialize one dataloader from caller-owned dataset indices."""
    subset = dataset.build_subset(indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def init_privacy_job(job: PrivacyJobConfig) -> tuple[torch.device, PrivacyDatasetHandle]:
    """Shared init for all privacy protocols: registries, device, dataset."""
    ensure_privacy_registries()
    use_gpu = job.options.gpu
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    dataset = load_privacy_dataset_wrapper(job)
    if job.target_model.source_type == TargetModelSourceType.CHECKPOINT and job.split_plan.strategy != PrivacySplitStrategy.LEGACY_COMPAT:
        raise ValueError("Checkpoint-backed privacy jobs require split_plan.strategy='legacy_compat'.")
    return device, dataset


def resolve_common_attack_extra_kwargs(
    job: PrivacyJobConfig,
    dataset: PrivacyDatasetHandle,
) -> dict[str, Any]:
    """Resolve num_classes validation and seed injection shared across protocols."""
    extra_attack_kwargs: dict[str, Any] = {}

    configured_num_classes = job.attack.attack_params.get("num_classes")
    if configured_num_classes is None:
        extra_attack_kwargs["num_classes"] = dataset.num_classes
    elif int(configured_num_classes) != int(dataset.num_classes):
        raise ValueError(
            f"Configured num_classes={configured_num_classes} does not match dataset num_classes={dataset.num_classes}."
        )

    if "seed" not in job.attack.attack_params:
        extra_attack_kwargs["seed"] = job.split_plan.seed

    return extra_attack_kwargs


def build_privacy_metadata(
    job: PrivacyJobConfig,
    loaded_target: LoadedPrivacyTargetModel,
    dataset: PrivacyDatasetHandle,
    split_plan: MaterializedPrivacySplits,
    device: torch.device,
    *,
    split_names: tuple[str, str, str, str] = ("shadow", "target_train", "target_val", "target_test"),
    split_strategy: str | None = None,
    task_attr: str | None = None,
    **extra_target_meta: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build target_metadata and dataset_metadata dicts shared by all protocols."""
    target_meta = {
        "model_id": job.target_model.model_id,
        "source_type": loaded_target.source_type.value,
        "checkpoint_path": str(loaded_target.checkpoint_path),
        "trained_now": loaded_target.trained_now,
        "training_recipe_id": loaded_target.training_recipe_id,
        "property_ratio": job.target_model.property_ratio,
        "property_name": job.target_model.property_name,
        "property_target_ratio": job.target_model.property_target_ratio,
        "device": str(device),
        **extra_target_meta,
    }
    s_name, tr_name, v_name, te_name = split_names
    dataset_meta = {
        "dataset_id": job.dataset.dataset_id,
        "num_classes": int(dataset.num_classes),
        "task_attr": task_attr if task_attr is not None else job.dataset.task_attr,
        "use_embeddings": bool(dataset.config.use_embeddings),
        "split_sizes": {
            s_name: len(split_plan.shadow),
            tr_name: len(split_plan.target_train),
            v_name: len(split_plan.target_val),
            te_name: len(split_plan.target_test),
        },
        "split_seed": int(job.split_plan.seed),
        "split_strategy": split_strategy if split_strategy is not None else job.split_plan.strategy.value,
        "dataset_seed": int(job.dataset.seed),
    }
    return target_meta, dataset_meta


def plan_privacy_split_indices(
    dataset_size: int,
    split_plan: PrivacySplitPlanConfig,
) -> MaterializedPrivacySplits:
    """Create deterministic shadow/target/train/val/test splits."""
    if dataset_size < 4:
        raise ValueError("dataset_size must be >= 4 for shadow/train/val/test splits.")
    gen = torch.Generator().manual_seed(split_plan.seed)
    perm = torch.randperm(dataset_size, generator=gen).tolist()
    shadow_sz = max(1, min(dataset_size - 3, int(dataset_size * split_plan.shadow_ratio)))
    pool = perm[:dataset_size - shadow_sz]
    shadow = perm[dataset_size - shadow_sz:]
    train_sz = min(max(1, int(len(pool) * split_plan.target_train_ratio)), len(pool) - 2)
    remainder = len(pool) - train_sz
    val_sz = max(1, remainder // 2)
    return MaterializedPrivacySplits(
        shadow=[int(i) for i in shadow],
        target_train=[int(i) for i in pool[:train_sz]],
        target_val=[int(i) for i in pool[train_sz:train_sz + val_sz]],
        target_test=[int(i) for i in pool[train_sz + val_sz:]],
    )


def resolve_privacy_protocol(job: PrivacyJobConfig) -> PrivacyProtocolId:
    """Resolve the privacy protocol from registered attack metadata."""
    info = AttackFactory.get_info(job.attack.attack_id)
    return PrivacyProtocolId(str(getattr(info.privacy_type, "value", info.privacy_type)))


def build_privacy_attack_kwargs(
    job: PrivacyJobConfig,
    *,
    model: torch.nn.Module,
    device: torch.device,
    task,
    verbose: bool = False,
    extra_attack_kwargs: dict | None = None,
) -> dict:
    """Build kwargs for AttackFactory.create(...)."""
    params = dict(job.attack.attack_params)
    if extra_attack_kwargs:
        params.update(extra_attack_kwargs)
    return {
        "class_id": job.attack.attack_id,
        "model": model,
        "device": device,
        "task": task,
        "verbose": verbose,
        **params,
    }
