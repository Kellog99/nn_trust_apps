"""Privacy result model and reusable metric helpers.

Includes Pydantic model for execution results, artifacts, and pending payloads,
plus metric-computation helpers for membership inference, property inference,
reconstruction, model inversion, and other common privacy attacks. Plugs into
the nn_trust privacy benchmarking framework's result serialization and analysis
pipeline.
"""

from collections.abc import Sequence
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PrivacyArtifactReference(BaseModel):
    artifact_id: str
    path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PrivacyPendingArtifactFormat(str, Enum):
    JSON = "json"
    PICKLE = "pickle"
    TORCH = "torch"
    TEXT = "text"
    IMAGE = "image"


class PrivacyPendingArtifact(BaseModel):
    artifact_id: str
    filename: str
    format: PrivacyPendingArtifactFormat
    payload: Any
    metadata: dict[str, Any] = Field(default_factory=dict)


class PrivacyExecutionResult(BaseModel):
    raw_outputs: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[PrivacyArtifactReference] = Field(default_factory=list)
    artifact_payloads: dict[str, Any] = Field(default_factory=dict, exclude=True)
    pending_artifacts: list[PrivacyPendingArtifact] = Field(default_factory=list, exclude=True)
    attack_metadata: dict[str, Any] = Field(default_factory=dict)
    target_metadata: dict[str, Any] = Field(default_factory=dict)
    dataset_metadata: dict[str, Any] = Field(default_factory=dict)


def _mean(values: Sequence[float]) -> float:
    return sum(float(v) for v in values) / len(values) if values else 0.0


def compute_membership_inference_metrics(
    member_scores: Sequence[float],
    non_member_scores: Sequence[float],
    *,
    threshold: float = 0.5,
    low_fpr_targets: Sequence[float] = (0.01, 0.05, 0.1, 0.2),
) -> dict[str, Any]:
    n_mem, n_non = len(member_scores), len(non_member_scores)
    preds = [1 if s >= threshold else 0 for s in member_scores]
    non_preds = [1 if s >= threshold else 0 for s in non_member_scores]
    tp = sum(preds)
    tn = sum(1 - int(p) for p in non_preds)
    total = n_mem + n_non
    tpr = tp / n_mem if n_mem else 0.0
    tnr = tn / n_non if n_non else 0.0
    rates = [v for v in (tpr, tnr) if n_mem and n_non]

    auc = None
    tpr_at_fpr = None
    if n_mem and n_non:
        try:
            from sklearn.metrics import roc_auc_score, roc_curve
            import numpy as np
            y_true = [1] * n_mem + [0] * n_non
            y_score = list(member_scores) + list(non_member_scores)
            auc = float(roc_auc_score(y_true, y_score))
            fpr, tpr_vals, _ = roc_curve(y_true, y_score)
            tpr_at_fpr = {}
            for target in sorted(float(r) for r in low_fpr_targets):
                idx = max(int(np.searchsorted(fpr, target, side="right")) - 1, 0)
                tpr_at_fpr[f"{target:.4g}"] = float(tpr_vals[idx])
        except ImportError:
            pass

    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / total) if total else 0.0,
        "balanced_accuracy": float(sum(rates) / len(rates)) if rates else 0.0,
        "tpr": float(tpr),
        "tnr": float(tnr),
        "roc_auc": auc,
        "tpr_at_fpr": tpr_at_fpr,
        "num_members": n_mem,
        "num_non_members": n_non,
    }


def compute_property_inference_metrics(
    *,
    property_prediction: int,
    property_confidence: float | None,
    ground_truth_label: str | None = None,
) -> dict[str, Any]:
    label = (ground_truth_label or "").lower()
    if label == "low":
        gt = 0
    elif label == "high":
        gt = 1
    else:
        gt = None
    pred = int(property_prediction)
    return {
        "property_prediction": pred,
        "property_confidence": None if property_confidence is None else float(property_confidence),
        "inference_granularity": "model_level",
        "ground_truth": gt,
        "ground_truth_label": ground_truth_label.upper() if ground_truth_label else None,
        "correct": None if gt is None else pred == gt,
    }


def compute_reconstruction_metrics(
    *,
    reconstruction_scores: Sequence[float],
    reconstruction_data: Sequence[dict[str, Any]],
    inference_granularity: str = "per_sample",
) -> dict[str, Any]:
    scores = [float(s) for s in reconstruction_scores]
    if not scores:
        return {"avg_confidence": 0.0, "num_reconstructions": 0, "inference_granularity": inference_granularity}

    best = max(reconstruction_data, key=lambda r: float(r["confidence"])) if reconstruction_data else None
    worst = min(reconstruction_data, key=lambda r: float(r["confidence"])) if reconstruction_data else None
    return {
        "avg_confidence": _mean(scores),
        "min_confidence": min(scores),
        "max_confidence": max(scores),
        "num_reconstructions": len(scores),
        "inference_granularity": inference_granularity,
        "best_confidence": float(best["confidence"]) if best else None,
        "best_class": int(best["y_target"]) if best else None,
        "worst_confidence": float(worst["confidence"]) if worst else None,
        "worst_class": int(worst["y_target"]) if worst else None,
    }


def compute_model_inversion_metrics(
    agreement_scores: Sequence[float],
    victim_correct: Sequence[float],
    surrogate_correct: Sequence[float],
    *,
    query_budget_requested: int,
    query_budget_used: int,
    fit_runtime_sec: float,
    distillation_history: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    final_loss = None
    if distillation_history and distillation_history.get("loss"):
        final_loss = float(distillation_history["loss"][-1])
    return {
        "fidelity": _mean(agreement_scores),
        "surrogate_accuracy": _mean(surrogate_correct),
        "victim_accuracy": _mean(victim_correct),
        "query_budget_requested": int(query_budget_requested),
        "query_budget_used": int(query_budget_used),
        "fit_runtime_sec": float(fit_runtime_sec),
        "distillation_final_loss": final_loss,
        "num_eval_samples": len(agreement_scores),
    }
