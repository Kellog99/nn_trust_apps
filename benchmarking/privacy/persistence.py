"""Persistence helpers for app-side privacy execution results."""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import torch

from .job_models import PrivacyJobConfig, PrivacyProtocolId
from .metrics import (
    PrivacyArtifactReference,
    PrivacyExecutionResult,
    PrivacyPendingArtifact,
    PrivacyPendingArtifactFormat,
)

SUPPORTED_PRIVACY_OUTPUT_FORMATS = frozenset({"json", "pickle"})
_OUTPUT_FILE_BY_FORMAT = {
    "json": "result.json",
    "pickle": "result.pkl",
}
_RAW_OUTPUTS_FILENAME = "raw_outputs.pkl"


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_json(path: str | Path, payload: Any, indent: int | None = None) -> Path:
    p = Path(path)
    _ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent)
    return p



def _build_privacy_result_identity_payload(
    job: PrivacyJobConfig,
    protocol: PrivacyProtocolId,
) -> dict[str, Any]:
    """Build the stable config payload that defines privacy result identity."""
    return {
        "protocol": protocol.value,
        "dataset": job.dataset.model_dump(mode="json"),
        "split_plan": job.split_plan.model_dump(mode="json"),
        "target_model": job.target_model.model_dump(mode="json"),
        "attack": job.attack.model_dump(mode="json"),
        "artifact_options": {
            "num_images_to_save": int(job.options.num_images_to_save),
            "save_perturbation": bool(job.options.save_perturbation),
        },
    }


def compute_privacy_result_fingerprint(job: PrivacyJobConfig, protocol: PrivacyProtocolId) -> str:
    """Compute a stable fingerprint for one privacy result configuration."""
    identity_payload = _build_privacy_result_identity_payload(job, protocol)
    serialized_payload = json.dumps(identity_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()[:16]


def resolve_privacy_output_dir(job: PrivacyJobConfig, protocol: PrivacyProtocolId) -> Path:
    """Resolve the deterministic output directory for one privacy job."""
    fp = compute_privacy_result_fingerprint(job, protocol)
    return Path(job.options.output_path) / protocol.value / job.target_model.model_id / job.dataset.dataset_id / job.attack.attack_id / fp


def _candidate_privacy_result_paths(output_dir: Path) -> list[Path]:
    """Return all supported serialized result paths under one output directory."""
    return [output_dir / filename for filename in _OUTPUT_FILE_BY_FORMAT.values()]


def resolve_privacy_result_path(job: PrivacyJobConfig, protocol: PrivacyProtocolId) -> Path:
    output_format = job.options.output_format
    if output_format not in SUPPORTED_PRIVACY_OUTPUT_FORMATS:
        raise ValueError(f"Unsupported privacy output format '{output_format}'. Supported: {sorted(SUPPORTED_PRIVACY_OUTPUT_FORMATS)}.")
    return resolve_privacy_output_dir(job, protocol) / _OUTPUT_FILE_BY_FORMAT[output_format]


def resolve_existing_privacy_result_path(job: PrivacyJobConfig, protocol: PrivacyProtocolId) -> Path | None:
    """Resolve any existing serialized result path for one privacy job."""
    preferred_path = resolve_privacy_result_path(job, protocol)
    if preferred_path.exists():
        return preferred_path

    output_dir = resolve_privacy_output_dir(job, protocol)
    for candidate_path in _candidate_privacy_result_paths(output_dir):
        if candidate_path.exists():
            return candidate_path
    return None


def resolve_privacy_raw_outputs_path(job: PrivacyJobConfig, protocol: PrivacyProtocolId) -> Path:
    """Resolve the persisted raw-output artifact path for one privacy job."""
    return resolve_privacy_output_dir(job, protocol) / _RAW_OUTPUTS_FILENAME


def privacy_result_exists(job: PrivacyJobConfig, protocol: PrivacyProtocolId) -> bool:
    """Return whether a previously persisted privacy result exists for this job."""
    return resolve_existing_privacy_result_path(job, protocol) is not None


def _augment_artifacts_with_raw_outputs(
    result: PrivacyExecutionResult,
    *,
    raw_outputs_path: Path,
) -> list[PrivacyArtifactReference]:
    artifacts = list(result.artifacts)
    artifacts.append(
        PrivacyArtifactReference(
            artifact_id="raw_outputs",
            path=str(raw_outputs_path),
            metadata={"format": "pickle"},
        )
    )
    return artifacts


def _write_pending_artifact(output_dir: Path, artifact: PrivacyPendingArtifact) -> PrivacyArtifactReference:
    artifact_path = output_dir / artifact.filename

    if artifact.format == PrivacyPendingArtifactFormat.JSON:
        _write_json(artifact_path, artifact.payload, indent=2)
    elif artifact.format == PrivacyPendingArtifactFormat.PICKLE:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with open(artifact_path, "wb") as f:
            pickle.dump(artifact.payload, f)
    elif artifact.format == PrivacyPendingArtifactFormat.TORCH:
        torch.save(artifact.payload, artifact_path)
    elif artifact.format == PrivacyPendingArtifactFormat.TEXT:
        artifact_path.write_text(str(artifact.payload), encoding="utf-8")
    elif artifact.format == PrivacyPendingArtifactFormat.IMAGE:
        import base64
        image_data_b64 = artifact.payload.get("image_data", "") if isinstance(artifact.payload, dict) else ""
        if image_data_b64:
            artifact_path.write_bytes(base64.b64decode(image_data_b64))
        else:
            raise ValueError(f"IMAGE artifact '{artifact.artifact_id}' has no base64 image_data payload.")
    else:
        raise ValueError(f"Unsupported pending privacy artifact format '{artifact.format}'.")

    return PrivacyArtifactReference(
        artifact_id=artifact.artifact_id,
        path=str(artifact_path),
        metadata={
            "format": artifact.format.value,
            **artifact.metadata,
        },
    )


def _load_privacy_artifact_payload(reference: PrivacyArtifactReference) -> Any:
    """Load one persisted privacy artifact payload from disk."""
    if reference.path is None:
        raise FileNotFoundError(f"Artifact '{reference.artifact_id}' has no persisted path.")

    artifact_path = Path(reference.path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Persisted privacy artifact not found: {artifact_path}.")

    artifact_format = reference.metadata.get("format")
    if artifact_format == "pickle":
        # Privacy artifacts are loaded only from app-owned output directories written by this same
        # persistence layer. This remains a local trust boundary and intentionally accepts pickle.
        with open(artifact_path, "rb") as handle:
            return pickle.load(handle)
    if artifact_format == "json":
        with open(artifact_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if artifact_format == "torch":
        # Surrogate checkpoints and tensor bundles are app-emitted artifacts under the same local
        # trust boundary. weights_only=False is intentional until artifact formats are hardened.
        return torch.load(artifact_path, map_location="cpu", weights_only=False)
    if artifact_format == "text":
        return artifact_path.read_text(encoding="utf-8")
    if artifact_format == "image":
        return artifact_path.read_bytes()

    raise ValueError(
        f"Unsupported persisted privacy artifact format '{artifact_format}' for artifact '{reference.artifact_id}'."
    )


def save_privacy_execution_result(
    job: PrivacyJobConfig,
    protocol: PrivacyProtocolId,
    result: PrivacyExecutionResult,
) -> Path:
    """Persist one privacy execution result bundle to disk."""
    output_format = job.options.output_format
    if output_format not in SUPPORTED_PRIVACY_OUTPUT_FORMATS:
        raise ValueError(f"Unsupported privacy output format '{output_format}'. Supported: {sorted(SUPPORTED_PRIVACY_OUTPUT_FORMATS)}.")
    output_dir = _ensure_dir(resolve_privacy_output_dir(job, protocol))
    raw_outputs_path = resolve_privacy_raw_outputs_path(job, protocol)
    raw_outputs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_outputs_path, "wb") as f:
        pickle.dump(result.raw_outputs, f)

    pending_artifact_refs = [_write_pending_artifact(output_dir, artifact) for artifact in result.pending_artifacts]

    persisted_result = result.model_copy(deep=True)
    persisted_result.artifacts = _augment_artifacts_with_raw_outputs(
        result,
        raw_outputs_path=raw_outputs_path,
    ) + pending_artifact_refs
    persisted_result.artifact_payloads = {}
    persisted_result.pending_artifacts = []
    persisted_result.raw_outputs = {}

    _write_json(output_dir / "metrics.json", result.metrics, indent=2)
    _write_json(
        output_dir / "metadata.json",
        {
            "result_fingerprint": compute_privacy_result_fingerprint(job, protocol),
            "attack_metadata": persisted_result.attack_metadata,
            "target_metadata": persisted_result.target_metadata,
            "dataset_metadata": persisted_result.dataset_metadata,
            "artifacts": [artifact.model_dump(mode="json") for artifact in persisted_result.artifacts],
        },
        indent=2,
    )

    result_payload: Any = persisted_result.model_dump(mode="json" if output_format == "json" else "python")
    result_path = output_dir / _OUTPUT_FILE_BY_FORMAT[output_format]
    if output_format == "json":
        _write_json(result_path, result_payload, indent=2)
    else:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "wb") as f:
            pickle.dump(result_payload, f)

    return output_dir


def load_privacy_execution_result(
    job: PrivacyJobConfig,
    protocol: PrivacyProtocolId,
) -> PrivacyExecutionResult:
    """Load one previously persisted privacy result bundle."""
    result_path = resolve_existing_privacy_result_path(job, protocol)
    if result_path is None:
        raise FileNotFoundError(
            f"No persisted privacy result found under {resolve_privacy_output_dir(job, protocol)}."
        )

    if result_path.suffix == ".json":
        with open(result_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    elif result_path.suffix == ".pkl":
        # Serialized privacy results are only reloaded from app-owned output directories written by
        # this same executor. This is a local trust boundary and intentionally accepts pickle.
        with open(result_path, "rb") as handle:
            payload = pickle.load(handle)
    else:
        raise ValueError(f"Unsupported persisted privacy result suffix '{result_path.suffix}'.")

    raw_outputs_path = resolve_privacy_raw_outputs_path(job, protocol)
    if raw_outputs_path.exists():
        # Raw outputs are persisted locally by this same executor, under the same trust boundary as
        # the main result bundle and auxiliary artifacts.
        with open(raw_outputs_path, "rb") as handle:
            payload["raw_outputs"] = pickle.load(handle)

    result = PrivacyExecutionResult(**payload)
    result.artifact_payloads = {
        artifact.artifact_id: _load_privacy_artifact_payload(artifact)
        for artifact in result.artifacts
        if artifact.artifact_id != "raw_outputs"
    }
    return result
