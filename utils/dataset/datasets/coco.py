from pathlib import Path
from typing import Optional

from torchvision.datasets import CocoDetection

from nn_trust.attack.utils.detection import LetterboxCocoTransform


class CocoDetectionDataset(CocoDetection):
    def __init__(
            self,
            root: str | Path,
            ann_file: str | Path,
            name: Optional[str] = None,
            new_shape: tuple[int, int] = (640, 640),
    ):
        raw_dataset = CocoDetection(root=str(root), annFile=str(ann_file))
        cat_id_to_label = {
            cat_id: idx
            for idx, cat_id in enumerate(sorted(raw_dataset.coco.getCatIds()))
        }
        super().__init__(
            root=str(root),
            annFile=str(ann_file),
            transforms=LetterboxCocoTransform(cat_id_to_label, new_shape=new_shape),
        )
        self.data_root = str(root)
        self.name = name
        self.cat_id_to_label = cat_id_to_label
