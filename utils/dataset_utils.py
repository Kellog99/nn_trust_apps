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
        **kwargs,
) -> DataLoader:
    """
    Return the dataloader to use and the inverse transformation to use for displaying the images
    """

    if not os.path.exists(dataset_path):
        raise ValueError(f"The dataset --------{dataset_path} does not exists.")

    def check_valid_image(filename: str):
        try:
            with PILImage.open(filename) as img:
                img.verify()  # Verify that the image is valid
            return True
        except Exception:
            return False

    dataset = ImageDatasetFolder(
        dataset_path,
        transform=transform,
        is_valid_file=check_valid_image
    )

    dataset.name = name if name is not None else Path(dataset).name

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

    dataloader = DataLoader(
        subdataset,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
    )

    return dataloader
