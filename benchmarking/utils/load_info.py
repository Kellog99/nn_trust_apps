import json
from pathlib import Path
from pprint import pprint
from typing import TypeVar, Type

from models import ModelInfo, DatasetInfo

T = TypeVar("T", ModelInfo, DatasetInfo)


def load_info(
        entry: dict,
        info_cls: Type[T]
) -> T:
    """
    Build a ModelInfo/DatasetInfo from one entry of the config's "models"/"datasets"
    list. Two shapes are supported:

      - {"source_path": "..."}                -> info is read from <source_path>/info.json,
                                                   repository defaults to source_path.
      - {<info fields directly inline>, ...}   -> validated as-is; "repository" must be set.
    """
    source_path: str | None = entry.get("source_path", None) or entry.get("repository", None)
    if source_path is None:
        raise ValueError(
            f"Entry {entry.get('id', entry.get('name', '<unknown>'))} needs either a "
            f"'source_path' or an explicit 'repository'."
        )

    source_path: Path = Path(source_path).expanduser()
    info_json = source_path / "info.json"

    if not info_json.parent.exists():
        raise FileNotFoundError(f"The model's folder, {info_json.parent}, does not exist.")
    if not info_json.exists():
        raise FileNotFoundError(f"Expected an info.json under {source_path}, found none.")

    with open(info_json, "rb") as f:
        data = json.load(f)

    # The information given by the config has a higher role
    data |= entry
    info = info_cls.model_validate(data)

    if getattr(info, "repository", None) is None:
        info.repository = str(source_path)
    return info
