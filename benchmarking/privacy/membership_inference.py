"""App-side execution protocol for privacy membership-inference attacks.

Implements the membership-inference attack protocol for the nn_trust privacy
benchmarking framework.  Supports generic membership-inference attacks via
AttackFactory, integrates with app-side dataset and model loading, handles
device placement, computes execution metrics, and serializes results to the
expected PrivacyExecutionResult format.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader

from nn_trust import ModelAdapter
from nn_trust.attack import AttackFactory

from .contracts import PrivacyDatasetHandle, resolve_privacy_model_task

from .job_models import MaterializedPrivacySplits, PrivacyJobConfig, PrivacyProtocolId
from .loading import (
    build_privacy_attack_kwargs,
    build_privacy_loader,
    build_privacy_metadata,
    init_privacy_job,
    load_privacy_target_model,
    plan_privacy_split_indices,
    resolve_common_attack_extra_kwargs,
    resolve_shadow_model_id,
)
from .metrics import PrivacyExecutionResult, compute_membership_inference_metrics

_DEFAULT_MEMBERSHIP_EVAL_BATCH_SIZE = 128


def _resolve_membership_batch_size(job: PrivacyJobConfig) -> int:
    """Resolve the batch size used for app-side membership evaluation."""
    return int(job.attack.attack_params.get("shadow_batch_size", _DEFAULT_MEMBERSHIP_EVAL_BATCH_SIZE))


def _materialize_membership_loaders(
    dataset: PrivacyDatasetHandle,
    *,
    split_plan: MaterializedPrivacySplits,
    batch_size: int,
    device: torch.device,
) -> dict[str, DataLoader]:
    """Create target/shadow dataloaders for one membership-inference job."""
    return {
        "shadow": build_privacy_loader(dataset, split_plan.shadow, batch_size=batch_size, shuffle=False, device=device),
        "member": build_privacy_loader(dataset, split_plan.target_train, batch_size=batch_size, shuffle=False, device=device),
        "non_member": build_privacy_loader(
            dataset,
            split_plan.target_test,
            batch_size=batch_size,
            shuffle=False,
            device=device,
        ),
    }


def _resolve_membership_attack_extra_kwargs(
    job: PrivacyJobConfig,
    *,
    dataset: PrivacyDatasetHandle,
) -> dict[str, Any]:
    """Resolve app-owned attack kwargs for membership-inference attacks."""
    extra_attack_kwargs = resolve_common_attack_extra_kwargs(job, dataset)

    configured_shadow_model_id = job.attack.attack_params.get("shadow_model_id")
    resolved_shadow_model_id = resolve_shadow_model_id(job)
    if configured_shadow_model_id is None:
        extra_attack_kwargs["shadow_model_id"] = resolved_shadow_model_id
    elif configured_shadow_model_id != resolved_shadow_model_id:
        raise ValueError(
            "Configured shadow_model_id does not match the target-model provenance. "
            f"Expected '{resolved_shadow_model_id}', got '{configured_shadow_model_id}'."
        )

    return extra_attack_kwargs


def _create_membership_attack(
    job: PrivacyJobConfig,
    *,
    target_model: torch.nn.Module,
    dataset: PrivacyDatasetHandle,
    device: torch.device,
):
    """Instantiate one privacy membership-inference attack."""
    task = resolve_privacy_model_task(job.target_model.model_id)
    attack_kwargs = build_privacy_attack_kwargs(
        job,
        model=ModelAdapter(target_model),
        device=device,
        task=task,
        verbose=job.verbose,
        extra_attack_kwargs=_resolve_membership_attack_extra_kwargs(job, dataset=dataset),
    )
    return AttackFactory.create(**attack_kwargs)


def _collect_membership_scores(
    attack: Any,
    dataloader: DataLoader,
    *,
    device: torch.device,
) -> tuple[list[float], str]:
    """Collect membership scores for one evaluation split."""
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
    score_source: str | None = None
    split_scores: list[float] = []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        if score_fn is None:
            try:
                batch_scores = attack.infer_proba(x, y)
                score_fn = attack.infer_proba
                score_source = "infer_proba"
            except NotImplementedError:
                batch_scores = attack.generate(x, y)
                score_fn = attack.generate
                score_source = "generate"
        else:
            batch_scores = score_fn(x, y)

        split_scores.extend(float(score) for score in batch_scores.detach().cpu().reshape(-1).tolist())

    return split_scores, score_source or "generate"

def run_membership_inference_job(job: PrivacyJobConfig) -> PrivacyExecutionResult:
    """Execute one app-side membership-inference privacy job."""
    device, dataset = init_privacy_job(job)
    split_plan = plan_privacy_split_indices(len(dataset.full_dataset), job.split_plan)
    batch_size = _resolve_membership_batch_size(job)
    loaders = _materialize_membership_loaders(
        dataset,
        split_plan=split_plan,
        batch_size=batch_size,
        device=device,
    )
    loaded_target = load_privacy_target_model(
        job,
        dataset=dataset,
        split_plan=split_plan,
        device=device,
    )
    attack = _create_membership_attack(
        job,
        target_model=loaded_target.model,
        dataset=dataset,
        device=device,
    )

    attack.fit(loaders["shadow"], shadow_model_id=resolve_shadow_model_id(job))

    member_scores, member_score_source = _collect_membership_scores(attack, loaders["member"], device=device)
    non_member_scores, non_member_score_source = _collect_membership_scores(attack, loaders["non_member"], device=device)
    if member_score_source != non_member_score_source:
        raise RuntimeError(
            "Membership score collection must use a single public attack output path, "
            f"got member={member_score_source!r}, non_member={non_member_score_source!r}."
        )

    metrics = compute_membership_inference_metrics(member_scores, non_member_scores)

    target_meta, dataset_meta = build_privacy_metadata(
        job, loaded_target, dataset, split_plan, device,
        shadow_model_id=resolve_shadow_model_id(job),
    )
    return PrivacyExecutionResult(
        raw_outputs={
            "member_scores": member_scores,
            "non_member_scores": non_member_scores,
            "member_labels": [1] * len(member_scores),
            "non_member_labels": [0] * len(non_member_scores),
        },
        metrics=metrics,
        attack_metadata={
            "attack_id": job.attack.attack_id,
            "protocol": PrivacyProtocolId.MEMBERSHIP_INFERENCE.value,
            "score_source": member_score_source,
        },
        target_metadata=target_meta,
        dataset_metadata=dataset_meta,
    )
