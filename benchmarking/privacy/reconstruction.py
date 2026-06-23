"""App-side execution protocol for privacy reconstruction attacks."""

from typing import Any

import torch

from nn_trust import ModelAdapter
from nn_trust.attack import AttackFactory

from .contracts import PrivacyDatasetHandle, resolve_privacy_model_task

from .artifact_rendering import (
    build_reconstruction_gallery_html,
    can_render_reconstruction_record,
    encode_reconstruction_png_base64,
)
from .job_models import MaterializedPrivacySplits, PrivacyJobConfig, PrivacyProtocolId
from .loading import (
    build_privacy_attack_kwargs,
    build_privacy_metadata,
    init_privacy_job,
    load_privacy_target_model,
    plan_privacy_split_indices,
    resolve_common_attack_extra_kwargs,
)
from .metrics import (
    PrivacyExecutionResult,
    PrivacyPendingArtifact,
    PrivacyPendingArtifactFormat,
    compute_reconstruction_metrics,
)


def _resolve_reconstruction_attack_extra_kwargs(
    job: PrivacyJobConfig,
    *,
    dataset: PrivacyDatasetHandle,
) -> dict[str, Any]:
    """Resolve app-owned attack kwargs for reconstruction attacks."""
    extra_kwargs = resolve_common_attack_extra_kwargs(job, dataset)
    if "input_shape" not in job.attack.attack_params:
        extra_kwargs["input_shape"] = _infer_reconstruction_input_shape(dataset)
    return extra_kwargs


def _plan_reconstruction_split_indices(
    job: PrivacyJobConfig,
    dataset: PrivacyDatasetHandle,
) -> tuple[MaterializedPrivacySplits, str]:
    """Resolve the dataset split used for reconstruction target training.

    Fredrikson et al.'s Face-Rec experiment on AT&T/ORL faces trains on seven
    images per subject and validates on the remaining three. The dataset
    wrapper exposes that split explicitly; other reconstruction datasets retain
    the generic privacy split planner.
    """
    paper_splitter = getattr(dataset, "get_paper_train_validation_indices", None)
    if callable(paper_splitter):
        target_train, target_val = paper_splitter()
        return (
            MaterializedPrivacySplits(
                shadow=[],
                target_train=list(target_train),
                target_val=list(target_val),
                target_test=[],
            ),
            "att_faces_paper_7_3",
        )

    return (
        plan_privacy_split_indices(len(dataset.full_dataset), job.split_plan),
        job.split_plan.strategy.value,
    )


def _create_reconstruction_attack(
    job: PrivacyJobConfig,
    *,
    target_model: torch.nn.Module,
    dataset: PrivacyDatasetHandle,
    device: torch.device,
) -> "MIFaceAttack":
    """Instantiate one app-side reconstruction attack."""
    task = resolve_privacy_model_task(job.target_model.model_id)
    attack_kwargs = build_privacy_attack_kwargs(
        job,
        model=ModelAdapter(target_model),
        device=device,
        task=task,
        verbose=job.verbose,
        extra_attack_kwargs=_resolve_reconstruction_attack_extra_kwargs(job, dataset=dataset),
    )
    attack = AttackFactory.create(**attack_kwargs)
    if not isinstance(attack, "MIFaceAttack"):
        raise TypeError(
            "The current app-side reconstruction executor only supports MIFaceAttack, "
            f"got {type(attack).__name__}."
        )
    return attack


def _infer_reconstruction_input_shape(dataset: PrivacyDatasetHandle) -> tuple[int, ...]:
    """Infer the model input shape from one dataset sample."""
    sample = dataset.full_dataset[0]
    x = sample[0] if isinstance(sample, tuple | list) else sample
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return tuple(int(dim) for dim in x.shape)


def _collect_reconstruction_outputs(
    attack: "MIFaceAttack",
    *,
    num_classes: int,
    input_shape: tuple[int, ...],
    device: torch.device,
) -> tuple[list[float], list[int], list[dict[str, Any]]]:
    """Run one reconstruction per class label through the public attack API."""
    if num_classes <= 0:
        raise ValueError(f"Reconstruction requires a positive num_classes value, got {num_classes}.")

    target_labels = torch.arange(num_classes, device=device, dtype=torch.long)
    dummy_x = torch.zeros((num_classes, *input_shape), device=device)
    scores = attack.generate(dummy_x, target_labels)
    reconstruction_scores = [float(score) for score in scores.detach().cpu().reshape(-1).tolist()]
    reconstruction_records = attack.get_reconstruction_data()

    if len(reconstruction_records) > len(reconstruction_scores):
        raise RuntimeError(
            "Reconstruction attacks must not record more reconstructions than returned scores, "
            f"got {len(reconstruction_records)} reconstructions for {len(reconstruction_scores)} scores."
        )

    return reconstruction_scores, [int(label) for label in target_labels.cpu().tolist()], reconstruction_records


def _select_reconstruction_records_for_artifact(
    reconstruction_records: list[dict[str, Any]],
    *,
    num_images_to_save: int,
) -> list[dict[str, Any]]:
    """Select the reconstruction records that should be persisted as artifacts.

    Always includes both the best (highest confidence) and worst (lowest
    confidence) reconstructions regardless of ``num_images_to_save``, then
    fills remaining slots from the top-confidence end.
    """
    if num_images_to_save == 0 or not reconstruction_records:
        return []

    ordered_records = sorted(
        reconstruction_records,
        key=lambda item: float(item["confidence"]),
        reverse=True,
    )
    if num_images_to_save < 0:
        return ordered_records

    # Always include best and worst; if only 1 slot, take the best.
    selected_indices: set[int] = set()
    selected_indices.add(0)  # best
    if len(ordered_records) > 1 and num_images_to_save >= 2:
        selected_indices.add(len(ordered_records) - 1)  # worst

    # Fill remaining slots from top-confidence end.
    for i in range(len(ordered_records)):
        if len(selected_indices) >= num_images_to_save:
            break
        selected_indices.add(i)

    return [ordered_records[i] for i in sorted(selected_indices)]


def _build_reconstruction_pending_artifacts(
    reconstruction_records: list[dict[str, Any]],
    *,
    dataset_id: str,
    model_id: str,
    attack_id: str,
    num_images_to_save: int,
) -> list[PrivacyPendingArtifact]:
    """Build persisted tensor, PNG, and gallery artifacts for reconstruction outputs."""
    selected_records = _select_reconstruction_records_for_artifact(
        reconstruction_records,
        num_images_to_save=num_images_to_save,
    )
    if not selected_records:
        return []

    # Always save the full tensor bundle.
    all_records = sorted(
        reconstruction_records,
        key=lambda item: float(item["confidence"]),
        reverse=True,
    )
    best_record = all_records[0] if all_records else None
    worst_record = all_records[-1] if len(all_records) > 1 else None

    artifacts = [
        PrivacyPendingArtifact(
            artifact_id="reconstructions",
            filename="reconstructions.pt",
            format=PrivacyPendingArtifactFormat.TORCH,
            payload={
                "reconstructions": selected_records,
                "best": best_record,
                "worst": worst_record,
            },
            metadata={
                "num_reconstructions_saved": len(selected_records),
                "num_reconstructions_total": len(reconstruction_records),
                "best_confidence": float(best_record["confidence"]) if best_record else None,
                "best_class": int(best_record["y_target"]) if best_record else None,
                "worst_confidence": float(worst_record["confidence"]) if worst_record else None,
                "worst_class": int(worst_record["y_target"]) if worst_record else None,
            },
        )
    ]

    # Save individual PNG images for each selected reconstruction.
    if all(can_render_reconstruction_record(record) for record in selected_records):
        for idx, record in enumerate(selected_records):
            rank_label = _rank_label_for_record(record, best_record, worst_record)
            png_filename = f"reconstruction_class{record['y_target']}_{rank_label}.png"
            artifacts.append(
                PrivacyPendingArtifact(
                    artifact_id=f"reconstruction_image_{idx}",
                    filename=png_filename,
                    format=PrivacyPendingArtifactFormat.IMAGE,
                    payload={
                        "image_data": encode_reconstruction_png_base64(record["x_recon"]),
                        "target_class": int(record["y_target"]),
                        "confidence": float(record["confidence"]),
                        "rank_label": rank_label,
                    },
                    metadata={
                        "media_type": "image/png",
                        "target_class": int(record["y_target"]),
                        "confidence": float(record["confidence"]),
                        "rank_label": rank_label,
                    },
                )
            )

        gallery_title = f"Reconstruction Gallery: {attack_id}"
        gallery_subtitle = (
            f"Dataset={dataset_id} | Model={model_id} | Saved reconstructions={len(selected_records)}"
        )
        artifacts.append(
            PrivacyPendingArtifact(
                artifact_id="reconstruction_gallery",
                filename="reconstruction_gallery.html",
                format=PrivacyPendingArtifactFormat.TEXT,
                payload=build_reconstruction_gallery_html(
                    selected_records,
                    title=gallery_title,
                    subtitle=gallery_subtitle,
                ),
                metadata={
                    "media_type": "text/html",
                    "gallery_scope": "reconstruction",
                    "num_reconstructions_saved": len(selected_records),
                },
            )
        )

    return artifacts


def _rank_label_for_record(
    record: dict[str, Any],
    best: dict[str, Any] | None,
    worst: dict[str, Any] | None,
) -> str:
    """Return a human-readable rank label (best/worst/intermediate) for one record."""
    if best is not None and record is best:
        return "best"
    if worst is not None and record is worst:
        return "worst"
    return "intermediate"

def run_reconstruction_job(job: PrivacyJobConfig) -> PrivacyExecutionResult:
    """Execute one app-side reconstruction privacy job."""
    device, dataset = init_privacy_job(job)
    split_plan, split_strategy = _plan_reconstruction_split_indices(job, dataset)
    loaded_target = load_privacy_target_model(
        job,
        dataset=dataset,
        split_plan=split_plan,
        device=device,
    )
    attack = _create_reconstruction_attack(
        job,
        target_model=loaded_target.model,
        dataset=dataset,
        device=device,
    )

    input_shape = _infer_reconstruction_input_shape(dataset)
    attack.fit(input_shape=input_shape)
    reconstruction_scores, reconstruction_labels, reconstruction_records = _collect_reconstruction_outputs(
        attack,
        num_classes=int(dataset.num_classes),
        input_shape=input_shape,
        device=device,
    )
    metrics = compute_reconstruction_metrics(
        reconstruction_scores=reconstruction_scores,
        reconstruction_data=reconstruction_records,
        inference_granularity=attack.INFERENCE_GRANULARITY.value,
    )

    target_meta, dataset_meta = build_privacy_metadata(
        job, loaded_target, dataset, split_plan, device,
        split_strategy=split_strategy,
    )
    return PrivacyExecutionResult(
        raw_outputs={
            "reconstruction_scores": reconstruction_scores,
            "reconstruction_labels": reconstruction_labels,
        },
        metrics=metrics,
        pending_artifacts=_build_reconstruction_pending_artifacts(
            reconstruction_records,
            dataset_id=job.dataset.dataset_id,
            model_id=job.target_model.model_id,
            attack_id=job.attack.attack_id,
            num_images_to_save=int(job.options.num_images_to_save),
        ),
        attack_metadata={
            "attack_id": job.attack.attack_id,
            "protocol": PrivacyProtocolId.RECONSTRUCTION.value,
            "score_source": "generate",
        },
        target_metadata=target_meta,
        dataset_metadata=dataset_meta,
    )
