"""Prepare YOLOv8n and the annotated COCO 2017 validation split.

Run ``python -m utils.model.download_yolo`` to populate both repositories.
Existing checkpoints, images, and annotations are reused.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile


MODEL_NAME = "yolov8n.pt"
MODEL_DIR = Path(os.environ.get(
    "YOLO_MODEL_DIR", "~/Desktop/StableAI/model_repository/yolov8"
)).expanduser()
DATASET_DIR = Path(os.environ.get(
    "COCO_VAL_DIR", "~/Desktop/StableAI/dataset_repository/coco_val2017"
)).expanduser()
COCO_URL = "https://images.cocodataset.org"


def download_archive(url: str, destination: Path) -> Path:
    """Keep an archive locally so interrupted extraction can be retried."""
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        try:
            urlretrieve(url, partial)
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)
    return destination


def prepare_model(model_dir: Path) -> None:
    from ultralytics import YOLO

    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / "model.pt"
    if not checkpoint.is_file():
        yolo = YOLO(MODEL_NAME)
        source = getattr(yolo, "ckpt_path", None)
        if source is None or not Path(source).is_file():
            source = Path(MODEL_NAME)
        if not Path(source).is_file():
            raise FileNotFoundError(f"Could not find downloaded checkpoint for {MODEL_NAME}")
        shutil.copy2(source, checkpoint)

    info = {
        "model_type": "ultralytics",
        "name": "yolov8",
        "id": "yolov8",
        "num_classes": 80,
        "task": "detection",
        "domain": "computer_vision",
        "input_dimensionality": [3, 640, 640],
        "repository": str(model_dir.resolve()),
        "transformation": {
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "size": 640,
        },
    }
    (model_dir / "info.json").write_text(json.dumps(info, indent=4), encoding="utf-8")


def prepare_dataset(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    images_dir = dataset_dir / "val2017"
    annotations_file = dataset_dir / "annotations" / "instances_val2017.json"

    if not annotations_file.is_file():
        archive = download_archive(
            f"{COCO_URL}/annotations/annotations_trainval2017.zip",
            dataset_dir / "annotations_trainval2017.zip",
        )
        with ZipFile(archive) as zipped:
            annotations_file.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open("annotations/instances_val2017.json") as source:
                with annotations_file.open("wb") as target:
                    shutil.copyfileobj(source, target)

    with annotations_file.open(encoding="utf-8") as source:
        annotation = json.load(source)
    image_names = {image["file_name"] for image in annotation["images"]}
    missing = image_names - {path.name for path in images_dir.glob("*.jpg")}
    if missing:
        archive = download_archive(
            f"{COCO_URL}/zips/val2017.zip",
            dataset_dir / "val2017.zip",
        )
        with ZipFile(archive) as zipped:
            images_dir.mkdir(parents=True, exist_ok=True)
            for name in sorted(missing):
                with zipped.open(f"val2017/{name}") as source:
                    with (images_dir / name).open("wb") as target:
                        shutil.copyfileobj(source, target)

    info = {
        "id": "coco_val2017",
        "name": "coco_val2017",
        "task": "detection",
        "domain": "computer_vision",
        "dataset_type": "coco",
        "num_classes": 80,
        "input_dimensionality": [3, 640, 640],
        "repository": str(dataset_dir.resolve()),
        "num_samples": len(image_names),
        "batch_size": 1,
        "num_workers": 1,
        "images_dir": "val2017",
        "annotations_file": "annotations/instances_val2017.json",
    }
    (dataset_dir / "info.json").write_text(json.dumps(info, indent=4), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    args = parser.parse_args()
    prepare_model(args.model_dir.expanduser())
    prepare_dataset(args.dataset_dir.expanduser())


if __name__ == "__main__":
    main()
