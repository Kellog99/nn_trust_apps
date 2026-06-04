"""App-side execution protocol for privacy property-inference attacks."""

from typing import Any

import torch
from torch.utils.data import DataLoader

from nn_trust import ModelAdapter
from nn_trust.attack import AttackFactory
from nn_trust.attack.privacy import PropertyInferenceAttack, PropertyInferenceConfig

from .contracts import (
    PrivacyAttributeDatasetHandle,
    PrivacyDatasetHandle,
    get_privacy_binary_attribute_values,
    resolve_privacy_model_task,
)

from .job_models import PrivacyJobConfig, PrivacyProtocolId
from .loading import (
    build_privacy_metadata,
    init_privacy_job,
    load_privacy_target_model,
    resolve_common_attack_extra_kwargs,
    resolve_shadow_model_id,
)
from .metrics import PrivacyExecutionResult, compute_property_inference_metrics
from .protocol import build_privacy_attack_kwargs
from .split_planning import MaterializedPrivacySplits, plan_privacy_split_indices

_PROPERTY_INFERENCE_DEFAULTS = PropertyInferenceConfig
_DEFAULT_PROPERTY_INFERENCE_BATCH_SIZE = int(
    _PROPERTY_INFERENCE_DEFAULTS.model_fields["shadow_batch_size"].default
)
_DEFAULT_PROPERTY_NAME = str(_PROPERTY_INFERENCE_DEFAULTS.model_fields["property_name"].default)


def _resolve_property_inference_batch_size(job: PrivacyJobConfig) -> int:
    """Resolve the batch size used to materialize property-aware shadow batches."""
    return int(job.attack.attack_params.get("shadow_batch_size", _DEFAULT_PROPERTY_INFERENCE_BATCH_SIZE))


def _resolve_property_name(job: PrivacyJobConfig) -> str:
    """Resolve the inferred property name for shadow-batch materialization."""
    configured_property_name = job.attack.attack_params.get("property_name")
    target_property_name = job.target_model.property_name

    if configured_property_name is None and target_property_name is None:
        return _DEFAULT_PROPERTY_NAME
    if configured_property_name is None:
        return str(target_property_name)
    if target_property_name is None:
        return str(configured_property_name)
    if str(configured_property_name) != str(target_property_name):
        raise ValueError(
            "Configured property_name does not match target-model provenance. "
            f"Expected '{target_property_name}', got '{configured_property_name}'."
        )
    return str(configured_property_name)


def _resolve_property_inference_task_attr(job: PrivacyJobConfig) -> str | None:
    """Resolve the task attribute used by the target and shadow models."""
    configured_task_attr = job.attack.attack_params.get("task_attr")
    dataset_task_attr = job.dataset.task_attr

    if configured_task_attr is None:
        return dataset_task_attr
    if dataset_task_attr is None:
        return str(configured_task_attr)
    if str(configured_task_attr) != dataset_task_attr:
        raise ValueError(
            "Configured attack task_attr does not match dataset.task_attr. "
            f"Expected '{dataset_task_attr}', got '{configured_task_attr}'."
        )
    return str(configured_task_attr)


def _build_property_inference_shadow_batches(
    dataset: PrivacyAttributeDatasetHandle,
    *,
    split_plan: MaterializedPrivacySplits,
    property_name: str,
    batch_size: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Materialize property-aware shadow batches for the app-side fit call."""
    shadow_indices = list(split_plan.shadow)
    shadow_loader = DataLoader(
        dataset.build_subset(shadow_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    attribute_values = get_privacy_binary_attribute_values(
        dataset,
        attribute_name=property_name,
        indices=shadow_indices,
    )
    if len(attribute_values) != len(shadow_indices):
        raise RuntimeError(
            "Property-attribute materialization must align with the app-owned shadow split, "
            f"got {len(attribute_values)} attribute values for {len(shadow_indices)} shadow indices."
        )

    shadow_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    batch_start = 0
    for x, y in shadow_loader:
        batch_end = batch_start + len(x)
        attribute_batch = attribute_values[batch_start:batch_end]
        shadow_batches.append((x, y, attribute_batch))
        batch_start = batch_end

    return shadow_batches


def _resolve_property_inference_attack_extra_kwargs(
    job: PrivacyJobConfig,
    *,
    dataset: PrivacyDatasetHandle,
) -> dict[str, Any]:
    """Resolve app-owned attack kwargs for property-inference attacks."""
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

    resolved_task_attr = _resolve_property_inference_task_attr(job)
    if resolved_task_attr is not None and "task_attr" not in job.attack.attack_params:
        extra_attack_kwargs["task_attr"] = resolved_task_attr

    return extra_attack_kwargs


def _require_attribute_capable_dataset(dataset: PrivacyDatasetHandle) -> PrivacyAttributeDatasetHandle:
    """Validate that the loaded dataset exposes the public attribute-access API."""
    getter = getattr(dataset, "get_binary_attribute_values", None)
    if not callable(getter):
        raise TypeError(
            "Property inference requires a privacy dataset exposing public binary attribute access, "
            f"got {type(dataset).__name__}."
        )
    return dataset


def _create_property_inference_attack(
    job: PrivacyJobConfig,
    *,
    target_model: torch.nn.Module,
    dataset: PrivacyDatasetHandle,
    device: torch.device,
) -> PropertyInferenceAttack:
    """Instantiate one app-side property-inference attack."""
    task = resolve_privacy_model_task(job.target_model.model_id)
    attack_kwargs = build_privacy_attack_kwargs(
        job,
        model=ModelAdapter(target_model),
        device=device,
        task=task,
        verbose=job.verbose,
        extra_attack_kwargs=_resolve_property_inference_attack_extra_kwargs(job, dataset=dataset),
    )
    attack = AttackFactory.create(**attack_kwargs)
    if not isinstance(attack, PropertyInferenceAttack):
        raise TypeError(
            "The current app-side property-inference executor only supports PropertyInferenceAttack, "
            f"got {type(attack).__name__}."
        )
    return attack


def _infer_target_property(
    attack: PropertyInferenceAttack,
    *,
    device: torch.device,
) -> tuple[int, float | None, str]:
    """Run one model-level property inference using the public attack API."""
    dummy_x = torch.zeros(1, device=device)
    dummy_y = torch.zeros(1, dtype=torch.long, device=device)

    prediction_tensor = attack.generate(dummy_x, dummy_y)
    prediction = int(prediction_tensor.detach().cpu().reshape(-1)[0].item())

    try:
        confidence_tensor = attack.infer_proba(dummy_x, dummy_y)
    except NotImplementedError:
        return prediction, None, "generate"

    confidence = float(confidence_tensor.detach().cpu().reshape(-1)[0].item())
    return prediction, confidence, "infer_proba"

def run_property_inference_job(job: PrivacyJobConfig) -> PrivacyExecutionResult:
    """Execute one app-side property-inference privacy job."""
    device, raw_dataset = init_privacy_job(job)
    dataset = _require_attribute_capable_dataset(raw_dataset)
    split_plan = plan_privacy_split_indices(len(dataset.full_dataset), job.split_plan)
    property_name = _resolve_property_name(job)
    shadow_batches = _build_property_inference_shadow_batches(
        dataset,
        split_plan=split_plan,
        property_name=property_name,
        batch_size=_resolve_property_inference_batch_size(job),
        device=device,
    )
    loaded_target = load_privacy_target_model(
        job,
        dataset=dataset,
        split_plan=split_plan,
        device=device,
    )
    attack = _create_property_inference_attack(
        job,
        target_model=loaded_target.model,
        dataset=dataset,
        device=device,
    )
    if (
        job.target_model.source_type.value == "train"
        and "shadow_subset_size" not in job.attack.attack_params
        and hasattr(attack.config, "shadow_subset_size")
    ):
        attack.config.shadow_subset_size = len(split_plan.target_train)

    attack.fit(shadow_batches, shadow_model_id=resolve_shadow_model_id(job))
    property_prediction, property_confidence, score_source = _infer_target_property(attack, device=device)

    metrics = compute_property_inference_metrics(
        property_prediction=property_prediction,
        property_confidence=property_confidence,
        ground_truth_label=job.target_model.property_ratio,
    )

    target_meta, dataset_meta = build_privacy_metadata(
        job, loaded_target, dataset, split_plan, device,
        task_attr=attack.config.task_attr,
    )
    return PrivacyExecutionResult(
        raw_outputs={
            "property_prediction": [property_prediction],
            "property_confidence": [] if property_confidence is None else [property_confidence],
        },
        metrics=metrics,
        attack_metadata={
            "attack_id": job.attack.attack_id,
            "protocol": PrivacyProtocolId.PROPERTY_INFERENCE.value,
            "score_source": score_source,
            "feature_strategy": "baseline",
            "meta_classifier_type": "deepsets",
            "property_name": property_name,
            "shadow_subset_size": attack.config.shadow_subset_size,
            "shadow_model_id": resolve_shadow_model_id(job),
        },
        target_metadata=target_meta,
        dataset_metadata=dataset_meta,
    )
