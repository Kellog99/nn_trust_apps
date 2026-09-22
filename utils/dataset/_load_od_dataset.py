from pathlib import Path
from typing import Optional

from utils.dataset.datasets.coco import CocoDetectionDataset


def _load_coco(
        root: Path,
        model_type: Optional[str] = None,
        images_dir: Optional[str] = None,
        annotations_file: Optional[str] = None,
        new_shape: tuple[int, int] = (640, 640),
        **kwargs,
) -> CocoDetectionDataset:
    if model_type != "ultralytics":
        raise ValueError("The current detection dataloader supports only ultralytics models.")
    return CocoDetectionDataset(
        root=root / (images_dir if images_dir is not None else "val2017"),
        ann_file=root / (annotations_file if annotations_file is not None
                         else "annotations/instances_val2017.json"),
        new_shape=new_shape,
    )
