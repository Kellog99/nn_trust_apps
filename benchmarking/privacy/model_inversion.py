"""App-side execution protocol for privacy model-inversion attacks.

Implements the model inversion attack protocol for the nn_trust privacy
benchmarking framework.  Supports generic model inversion attacks via
AttackFactory, integrates with app-side dataset and model loading, handles
device placement, computes execution metrics, and serializes results to the
expected PrivacyExecutionResult format.
"""

import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nn_trust import ModelAdapter
from nn_trust.attack import AttackFactory
from nn_trust.attack.privacy import E3DataDependentAttack

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
    resolve_surrogate_model_id,
)
from .metrics import (
    PrivacyExecutionResult,
    PrivacyPendingArtifact,
    PrivacyPendingArtifactFormat,
    compute_model_inversion_metrics,
)

_DEFAULT_EXTRACTION_EVAL_BATCH_SIZE = 256


def _resolve_extraction_batch_size(job: PrivacyJobConfig) -> int:
    """Resolve the batch size used for extraction fit/evaluation."""
    return int(job.attack.attack_params.get("extract_batch_size", _DEFAULT_EXTRACTION_EVAL_BATCH_SIZE))


def _materialize_extraction_loaders(
    dataset: PrivacyDatasetHandle,
    *,
    split_plan: MaterializedPrivacySplits,
    batch_size: int,
    device: torch.device,
) -> dict[str, DataLoader]:
    """Create query/evaluation loaders for one extraction job."""
    return {
        "query_pool": build_privacy_loader(
            dataset,
            split_plan.shadow,
            batch_size=batch_size,
            shuffle=False,
            device=device,
        ),
        "evaluation": build_privacy_loader(
            dataset,
            split_plan.target_test,
            batch_size=batch_size,
            shuffle=False,
            device=device,
        ),
    }


def _resolve_model_inversion_attack_extra_kwargs(
    job: PrivacyJobConfig,
    *,
    dataset: PrivacyDatasetHandle,
) -> dict[str, Any]:
    """Resolve app-owned attack kwargs for extraction attacks."""
    extra_attack_kwargs = resolve_common_attack_extra_kwargs(job, dataset)

    if "surrogate_model_id" not in job.attack.attack_params:
        extra_attack_kwargs["surrogate_model_id"] = resolve_surrogate_model_id(job)

    return extra_attack_kwargs


def _create_model_inversion_attack(
    job: PrivacyJobConfig,
    *,
    target_model: torch.nn.Module,
    dataset: PrivacyDatasetHandle,
    device: torch.device,
) -> E3DataDependentAttack:
    """Instantiate one app-side model-inversion attack."""
    task = resolve_privacy_model_task(job.target_model.model_id)
    attack_kwargs = build_privacy_attack_kwargs(
        job,
        model=ModelAdapter(target_model),
        device=device,
        task=task,
        verbose=job.verbose,
        extra_attack_kwargs=_resolve_model_inversion_attack_extra_kwargs(job, dataset=dataset),
    )
    attack = AttackFactory.create(**attack_kwargs)
    if not isinstance(attack, E3DataDependentAttack):
        raise TypeError(
            "The current app-side model-inversion executor only supports E3DataDependentAttack, "
            f"got {type(attack).__name__}."
        )
    return attack


def _collect_extraction_outputs(
    attack: E3DataDependentAttack,
    evaluation_loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, list[float]]:
    """Collect fidelity and accuracy signals on the app-owned evaluation split."""
    agreement_scores: list[float] = []
    victim_correct: list[float] = []
    surrogate_correct: list[float] = []
    victim_confidences: list[float] = []
    surrogate_confidences: list[float] = []

    victim = attack.config.model
    surrogate = attack.surrogate_model
    if surrogate is None:
        raise ValueError("Attack not fitted. Call fit() first.")

    victim.eval()
    surrogate.eval()
    surrogate.to(device)

    for x, y in evaluation_loader:
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            victim_logits = victim(x)
            surrogate_logits = surrogate(x)

            victim_probs = F.softmax(victim_logits, dim=1)
            surrogate_probs = F.softmax(surrogate_logits, dim=1)

            victim_pred = victim_probs.argmax(dim=1)
            surrogate_pred = surrogate_probs.argmax(dim=1)

            victim_correct_batch = (victim_pred == y).float()
            surrogate_correct_batch = (surrogate_pred == y).float()

            victim_conf = victim_probs.max(dim=1)[0]
            surrogate_conf = surrogate_probs.max(dim=1)[0]

        agreement = attack.generate(x, y)
        agreement_scores.extend(float(v) for v in agreement.detach().cpu().reshape(-1).tolist())
        victim_correct.extend(float(v) for v in victim_correct_batch.detach().cpu().reshape(-1).tolist())
        surrogate_correct.extend(float(v) for v in surrogate_correct_batch.detach().cpu().reshape(-1).tolist())
        victim_confidences.extend(float(v) for v in victim_conf.detach().cpu().reshape(-1).tolist())
        surrogate_confidences.extend(float(v) for v in surrogate_conf.detach().cpu().reshape(-1).tolist())

    return {
        "agreement_scores": agreement_scores,
        "victim_correct": victim_correct,
        "surrogate_correct": surrogate_correct,
        "victim_confidences": victim_confidences,
        "surrogate_confidences": surrogate_confidences,
    }


def _build_extraction_pending_artifacts(attack: E3DataDependentAttack) -> list[PrivacyPendingArtifact]:
    """Build app-side artifact payloads for one E3 extraction run."""
    surrogate = attack.surrogate_model
    if surrogate is None:
        return []

    checkpoint = {
        "model_state_dict": surrogate.state_dict(),
        "surrogate_model_id": attack.surrogate_model_id,
        "num_classes": attack.config.num_classes,
    }

    return [
        PrivacyPendingArtifact(
            artifact_id="surrogate_checkpoint",
            filename="extracted_surrogate.pt",
            format=PrivacyPendingArtifactFormat.TORCH,
            payload=checkpoint,
            metadata={
                "surrogate_model_id": attack.surrogate_model_id,
                "num_classes": int(attack.config.num_classes),
            },
        ),
    ]

def run_model_inversion_job(job: PrivacyJobConfig) -> PrivacyExecutionResult:
    """Execute one app-side model-inversion privacy job."""
    device, dataset = init_privacy_job(job)
    split_plan = plan_privacy_split_indices(len(dataset.full_dataset), job.split_plan)
    batch_size = _resolve_extraction_batch_size(job)
    loaders = _materialize_extraction_loaders(
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
    attack = _create_model_inversion_attack(
        job,
        target_model=loaded_target.model,
        dataset=dataset,
        device=device,
    )

    resolved_surrogate_model_id = resolve_surrogate_model_id(job)
    start = time.time()
    attack.fit(loaders["query_pool"], shadow_model_id=resolved_surrogate_model_id)
    fit_runtime_sec = time.time() - start

    extraction_outputs = _collect_extraction_outputs(
        attack,
        loaders["evaluation"],
        device=device,
    )
    metrics = compute_model_inversion_metrics(
        extraction_outputs["agreement_scores"],
        extraction_outputs["victim_correct"],
        extraction_outputs["surrogate_correct"],
        query_budget_requested=int(job.attack.attack_params.get("query_budget", attack.config.query_budget)),
        query_budget_used=int(job.attack.attack_params.get("query_budget", attack.config.query_budget)),
        fit_runtime_sec=fit_runtime_sec,
    )

    target_meta, dataset_meta = build_privacy_metadata(
        job, loaded_target, dataset, split_plan, device,
        split_names=("query_pool", "target_train", "target_val", "evaluation"),
    )
    return PrivacyExecutionResult(
        raw_outputs=extraction_outputs,
        metrics=metrics,
        pending_artifacts=_build_extraction_pending_artifacts(attack),
        attack_metadata={
            "attack_id": job.attack.attack_id,
            "protocol": PrivacyProtocolId.MODEL_INVERSION.value,
            "surrogate_model_id": resolved_surrogate_model_id,
        },
        target_metadata=target_meta,
        dataset_metadata=dataset_meta,
    )
