import os
import random
from pathlib import Path
from typing import Optional, Any

import numpy
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image as PILImage
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import transforms
from nn_trust import Task
from nn_trust.attack.detection_utils import LetterboxCocoTransform
from torchvision.datasets import CocoDetection

from models.info import Transformation


class ImageDatasetFolder(torchvision.datasets.ImageFolder):
    def __init__(
            self,
            root: str,
            transform=None,
            target_transform=None,
            is_valid_file=None
    ):
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file
        )
        self.data_root = root

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = PILImage.open(path).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

class CocoDetectionDataset(CocoDetection):
    def __init__(
            self,
            root: str | Path,
            ann_file: str | Path,
            name: Optional[str] = None,
            new_shape: tuple[int, int] = (640, 640),
    ):
        raw_dataset = CocoDetection(
            root=str(root),
            annFile=str(ann_file),
        )

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


def get_transformation(transformation: Transformation):
    out = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=getattr(transformation, "mean", (0.5, 0.5, 0.5)),
            std=getattr(transformation, "std", (0.5, 0.5, 0.5))
        )
    ]
    if transformation.size is not None:
        out.append(transforms.Resize((transformation.size, transformation.size)))
    if transformation.crop is not None:
        out.append(transforms.CenterCrop(transformation.crop))
    return transforms.Compose(out)


def get_dataloader(
        dataset_path: str,
        batch: int,
        transform: T.Compose,
        subset: Optional[int] = None,
        num_workers: int = 4,
        name: Optional[str] = None,
        task: Optional[Task] = None,
        **kwargs,
) -> DataLoader:
    """
    Return the dataloader to use and the inverse transformation to use for displaying the images
    """
    dataset_path = Path(dataset_path).expanduser()

    if not os.path.exists(dataset_path):
        raise ValueError(f"The dataset --------{dataset_path} does not exists.")

    def check_valid_image(filename: str):
        try:
            with PILImage.open(filename) as img:
                img.verify()  # Verify that the image is valid
            return True
        except Exception:
            return False

    match task:
        case Task.Detection:
            dataset = CocoDetectionDataset(
                root=dataset_path / "val2017",
                ann_file=dataset_path / "annotations" / "instances_val2017.json",
            )
        case Task.Classification:
            dataset = ImageDatasetFolder(
                dataset_path,
                transform=transform,
                is_valid_file=check_valid_image
            )
        case _:
            raise NotImplementedError(f"{task} not supported yet.")

    dataset.name = name if name is not None else dataset_path.name

    if subset is None or subset < 0:
        subset = list(range(len(dataset)))
    else:
        subset = list(range(min(subset, len(dataset))))
    subdataset = Subset(dataset, subset)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(1234)

    dataloader_kwargs = {
        "batch_size": batch,
        "shuffle": task != Task.Detection,
        "num_workers": num_workers,
        "worker_init_fn": seed_worker,
        "generator": g,
        "pin_memory": True,
    }

    if task == Task.Detection:
        dataloader_kwargs["collate_fn"] = lambda batch: tuple(zip(*batch))

    dataloader = DataLoader(subdataset, **dataloader_kwargs)

    return dataloader
