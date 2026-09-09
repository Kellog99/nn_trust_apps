from pathlib import Path
import json
import shutil
from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"

model_dir = Path("benchmark_assets/models/yolov8")
model_dir.mkdir(parents=True, exist_ok=True)

yolo = YOLO(MODEL_NAME)

ckpt_path = getattr(yolo, "ckpt_path", None)
if ckpt_path is not None and Path(ckpt_path).exists():
    shutil.copy2(ckpt_path, model_dir / "model.pt")
elif Path(MODEL_NAME).exists():
    shutil.copy2(MODEL_NAME, model_dir / "model.pt")
else:
    raise FileNotFoundError(f"Could not find downloaded checkpoint for {MODEL_NAME}")

info = {
    "type": "ultralytics",
    "name": "yolov8",
    "id": "yolov8",
    "num_classes": 80,
    "task": "detection",
    "domain": "computer_vision",
    "input_dimensionality": [3, 640, 640],
    "repository": "benchmark_assets/models/yolov8",
    "transformation": {
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "size": 640
    }
}

with open(model_dir / "info.json", "w") as f:
    json.dump(info, f, indent=4)