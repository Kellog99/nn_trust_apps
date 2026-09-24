import csv
import io
import json
from pathlib import Path
from typing import Optional, Any

import pandas as pd
from PIL import Image as PILImage
from matplotlib import pyplot as plt


def show_parquet_image(
        root: Path,
        image_col: str = "image",
        iloc: int = 0,
        label: str = "bytes",
        save: bool = False,
        save_dir: str = "./"
) -> None:
    if root is None:
        raise ValueError("root cannot be None.")
    root = Path(root)
    if not root.is_file():
        raise ValueError("root must be a file.")
    if root.suffix.lower() != ".parquet":
        raise ValueError("root must be a parquet file.")

    # 1. Read the Parquet file
    df: pd.DataFrame = pd.read_parquet(root)

    # 2. Extract the binary image data for a single row
    if image_col not in df.columns:
        raise ValueError(f"{image_col} is not a valid column name.")
    if iloc < 0 or iloc >= len(df):
        raise IndexError(f"{iloc} is out of range; valid range is 0-{len(df) - 1}.")
    image_data: Any = df[image_col].iloc[iloc]
    if not isinstance(image_data, dict) or label not in image_data:
        raise ValueError(f"{label} is not a valid label.")
    image_bytes = image_data[label]
    if image_bytes is None:
        raise ValueError(f"The image at row {iloc} has no {label} data.")

    # 3. Convert bytes into a PIL Image and display or save it.
    with PILImage.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")

    plt.imshow(image)
    if save:
        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.axis("off")
        plt.savefig(
            fname=output_dir / f"image_{iloc}.png",
            bbox_inches="tight",
            pad_inches=0
        )
        plt.close()
    else:
        plt.axis("off")
        plt.show()


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
    """Resolve common dataset layouts such as ``root/{train,val,test,data}``."""
    if split:
        candidate = root / split
        if not candidate.is_dir():
            raise ValueError(f"Dataset split {split!r} does not exist below {root}.")
        return candidate
    for candidate_name in ("test", "val", "validation", "train", "data"):
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
