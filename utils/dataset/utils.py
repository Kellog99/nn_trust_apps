import csv
import json
from pathlib import Path
from typing import Optional

from PIL import Image as PILImage


def _read_labels(root: Path) -> Optional[dict[str, int]]:
    csv_path, json_path = root / "labels.csv", root / "labels.json"
    if csv_path.exists():
        with csv_path.open(newline="") as file:
            rows = list(csv.DictReader(file))
        if not rows or not {"file", "label"}.issubset(rows[0]):
            raise ValueError("labels.csv must contain 'file' and 'label' columns.")
        return {row["file"]: int(row["label"]) for row in rows}
    if json_path.exists():
        with json_path.open() as file:
            values = json.load(file)
        if not isinstance(values, dict):
            raise ValueError("labels.json must be a mapping from file name to label.")
        return {str(name): int(label) for name, label in values.items()}
    return None


def _resolve_dataset_root(
        root: Path,
        split: Optional[str] = None
) -> Path:
    """Resolve the common ``root/{train,val,test}`` layout."""
    if split:
        candidate = root / split
        if not candidate.is_dir():
            raise ValueError(f"Dataset split {split!r} does not exist below {root}.")
        return candidate
    for candidate_name in ("test", "val", "validation", "train"):
        candidate = root / candidate_name
        if candidate.is_dir():
            return candidate
    return root


def _is_valid_image(filename: str) -> bool:
    try:
        with PILImage.open(filename) as image:
            image.verify()
        return True
    except Exception:
        return False
