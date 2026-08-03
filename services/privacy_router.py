import base64
from pathlib import Path

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Response
from fastapi.responses import FileResponse
from pydantic import ValidationError

from services.lib.privacy_jobs import get, submit, to_privacy_job_config
from models import ServerConfig, config_field
from models.privacy import PrivacyArtifactRef, PrivacyAttackOutput, PrivacyAttackProps
from benchmarking.privacy.metrics import PrivacyExecutionResult, PrivacyPendingArtifactFormat
from benchmarking.privacy.persistence import resolve_privacy_output_dir
from benchmarking.privacy.loading import resolve_privacy_protocol

router = APIRouter(prefix="/privacy", tags=["jobs management", "jobs utils"])

_MEDIA_TYPES = {"image": "image/png", "json": "application/json", "text": "text/plain"}


def _media_type(meta: dict, path: str | None = None) -> str:
    if m := meta.get("media_type"):
        return m
    return _MEDIA_TYPES.get(meta.get("format", ""), "application/octet-stream")


def _resolve_job(job_id: str):
    """Return the completed job result or raise the appropriate HTTPException."""
    job = get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if not job.future.done():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' is still running.")
    if job.future.exception() is not None:
        raise HTTPException(status_code=500, detail=str(job.future.exception()))
    return job


@router.post("/run")
async def run_privacy_attack(
    body: PrivacyAttackProps = Body(...),
    config: ServerConfig = Depends(config_field(attr_name=None)),
    device: str = Query(default="cpu", description="Device to run the attack on."),
) -> dict[str, str]:
    try:
        cfg = to_privacy_job_config(body=body, config=config, device=device)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors()) from e
    return {"job_id": submit(cfg)}


@router.get("/status/{job_id}")
def get_privacy_status(job_id: str) -> dict[str, str]:
    if (job := get(job_id)) is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if not job.future.done():
        return {"status": "running"}
    return {"status": "failed" if job.future.exception() else "completed"}


@router.get("/result/{job_id}")
def get_privacy_result(job_id: str) -> PrivacyAttackOutput:
    job = _resolve_job(job_id)
    result: PrivacyExecutionResult = job.future.result()

    reconstructions = None
    payload = result.artifact_payloads.get("reconstructions") or next(
        (a.payload for a in result.pending_artifacts if a.artifact_id == "reconstructions"), None
    )
    if isinstance(payload, dict):
        records = payload.get("reconstructions")
        if isinstance(records, list):
            reconstructions = [r["x_recon"] for r in records if isinstance(r, dict) and "x_recon" in r]

    artifacts: list[PrivacyArtifactRef] = []
    for a in result.artifacts:
        artifacts.append(PrivacyArtifactRef(
            artifact_id=a.artifact_id,
            filename=Path(a.path).name if a.path else a.artifact_id,
            media_type=_media_type(a.metadata, a.path),
            metadata=dict(a.metadata),
        ))
    for a in result.pending_artifacts:
        artifacts.append(PrivacyArtifactRef(
            artifact_id=a.artifact_id,
            filename=a.filename,
            media_type=_media_type(a.metadata | {"format": a.format.value}),
            metadata=dict(a.metadata),
        ))

    return PrivacyAttackOutput(
        metrics=result.metrics,
        reconstructions=reconstructions,
        artifacts=artifacts,
        attack_metadata=result.attack_metadata,
        target_metadata=result.target_metadata,
        dataset_metadata=result.dataset_metadata,
    )


@router.get("/artifact/{job_id}/{artifact_id}")
def get_privacy_artifact(job_id: str, artifact_id: str):
    job = _resolve_job(job_id)
    result: PrivacyExecutionResult = job.future.result()

    # persisted artifacts
    for a in result.artifacts:
        if a.artifact_id == artifact_id and a.path and Path(a.path).exists():
            return FileResponse(path=a.path, media_type=_media_type(a.metadata, a.path))

    # pending artifacts
    pending = next((a for a in result.pending_artifacts if a.artifact_id == artifact_id), None)
    if pending is None:
        raise HTTPException(status_code=404, detail=f"Artifact '{artifact_id}' not found.")

    output_dir = resolve_privacy_output_dir(job.config, resolve_privacy_protocol(job.config))
    output_path = output_dir / pending.filename
    media = _media_type(pending.metadata | {"format": pending.format.value})

    if output_path.exists():
        return FileResponse(path=str(output_path), media_type=media)
    if pending.format == PrivacyPendingArtifactFormat.IMAGE:
        data = (pending.payload or {}).get("image_data") if isinstance(pending.payload, dict) else None
        if data:
            return Response(content=base64.b64decode(data), media_type=media)
    raise HTTPException(status_code=404, detail=f"Artifact '{artifact_id}' not found.")
