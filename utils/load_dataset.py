import random
from pathlib import Path
from typing import Callable, Optional

import numpy
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.transforms import transforms

from models.info import Transformation, DATASET_TYPES, DatasetInfo
from utils.dataset._load_classification_dataset import (
    _load_auto,
    _load_image_folder,
    _load_flat,
    _load_parquet
)

_LOADERS: dict[DATASET_TYPES, Callable[..., Dataset]] = {
    "auto": _load_auto,
    "image_folder": _load_image_folder,
    "flat": _load_flat,
    "parquet": _load_parquet,
}


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
        dataset_info: DatasetInfo,
        batch: int,
        transform: T.Compose,
        subset: Optional[int] = None,
        num_workers: int = 4,
        dataset_type: DATASET_TYPES = "auto",
        split: Optional[str] = None,
        **kwargs,
) -> DataLoader:
    """
    Return the dataloader to use and the inverse transformation to use for displaying the images
    """

    root: Path = Path(dataset_path).expanduser()
    if not root.exists():
        raise ValueError(f"The dataset {root} does not exist.")

    # getting the dataset loader
    try:
        loader = _LOADERS[dataset_type]
    except KeyError:
        raise ValueError(
            f"Unsupported dataset type: {dataset_type}. "
            f"Supported types: {sorted(_LOADERS.keys())}. "
            "For COCO, YOLO, video, medical volumes, or another custom "
            "format, pass a torch.utils.data.Dataset instance."
        ) from None
    dataset: Dataset = loader(root=root, transform=transform, split=split, **kwargs)

    if subset is None or subset < 0:
        indices = list(range(len(dataset)))
    else:
        indices = list(range(min(subset, len(dataset))))
    subdataset = Subset(dataset, indices)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(1234)

    dataloader = DataLoader(
        subdataset,
        batch_size=batch,
        shuffle=True,
        num_workers=max(0, num_workers),
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
    )
    return dataloader
