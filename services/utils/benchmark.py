import json
from pathlib import Path

from fastapi import HTTPException

from models import DatasetInfo, ModelInfo


def _load_resource_info(resource_id: str, repository: str, *, dataset: bool):
    """Resolve the ID-only representation used by the web client."""
    repository_root = Path(repository).expanduser().resolve()
    root = (repository_root / resource_id).resolve()
    try:
        root.relative_to(repository_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid resource ID")
    info_path = root / "info.json"
    if not info_path.is_file():
        raise HTTPException(status_code=404, detail=f"{'Dataset' if dataset else 'Model'} '{resource_id}' not found")

    with info_path.open("r", encoding="utf-8") as info_file:
        info = json.load(info_file)
    info["id"] = info.get("id", resource_id)
    info["name"] = info.get("name", resource_id)
    info["repository"] = str(root / "data" if dataset and (root / "data").is_dir() else root)
    if not dataset:
        # Older generated metadata calls this field `api`, while ModelInfo
        # uses `model_type`.
        info.setdefault("model_type", info.get("api", "plain"))
    return DatasetInfo.model_validate(info) if dataset else ModelInfo.model_validate(info)
