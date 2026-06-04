"""Dataset split planning utilities for privacy protocols.

Supports both generic shadow-pool-split and dataset-provided splits (e.g., for FRED results).
"""

from __future__ import annotations

import torch
from pydantic import BaseModel, Field

from .job_models import PrivacySplitPlanConfig


class MaterializedPrivacySplits(BaseModel):
    """Concrete dataset indices for one privacy execution."""

    shadow: list[int] = Field(default_factory=list)
    target_train: list[int] = Field(default_factory=list)
    target_val: list[int] = Field(default_factory=list)
    target_test: list[int] = Field(default_factory=list)

    @property
    def dataset_size(self) -> int:
        return sum(len(getattr(self, f)) for f in ("shadow", "target_train", "target_val", "target_test"))

    @property
    def target_pool_size(self) -> int:
        return len(self.target_train) + len(self.target_val) + len(self.target_test)


def _finalize(indices: list[int]) -> list[int]:
    return [int(i) for i in indices]


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

    train_sz = max(1, int(len(pool) * split_plan.target_train_ratio))
    train_sz = min(train_sz, len(pool) - 2)
    remainder = len(pool) - train_sz
    val_sz = max(1, remainder // 2)
    test_sz = remainder - val_sz

    return MaterializedPrivacySplits(
        shadow=_finalize(shadow),
        target_train=_finalize(pool[:train_sz]),
        target_val=_finalize(pool[train_sz:train_sz + val_sz]),
        target_test=_finalize(pool[train_sz + val_sz:]),
    )
