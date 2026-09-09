import random
from pathlib import Path
from typing import Callable, Optional

import numpy
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms as T

from models.info import Transformation, DATASET_TYPES, DatasetInfo, ParquetInfo
from utils.dataset._load_classification_dataset import (
    _load_image_folder,
    _load_flat,
    _load_parquet
)

_LOADERS: dict[DATASET_TYPES, Callable[..., Dataset]] = {
    "image_folder": _load_image_folder,
    "flat": _load_flat,
    "parquet": _load_parquet,
}


def get_transformation(transformation: Optional[Transformation] = None) -> T.Compose:
    out: list[Callable[..., object]] = [T.ToTensor()]
    if transformation is not None:
        out.append(
            T.Normalize(
                mean=getattr(transformation, "mean", (0.5, 0.5, 0.5)),
                std=getattr(transformation, "std", (0.5, 0.5, 0.5)),
            )
        )
        if transformation.size is not None:
            out.append(T.Resize((transformation.size, transformation.size)))
        if transformation.crop is not None:
            out.append(T.CenterCrop(transformation.crop))
    return T.Compose(out)


def get_dataloader(
        dataset_path: str | Path | None,
        batch: int,
        dataset_info: Optional[DatasetInfo] = None,
        transform: Optional[T.Compose] = None,
        subset: Optional[int] = None,
        num_workers: int = 4,
        dataset_type: DATASET_TYPES = "image_folder",
        folder_data: Optional[str] = None,
        parquet_info: Optional[ParquetInfo] = None,
        **kwargs,
) -> DataLoader:
    """
    Return the DataLoader to use and the inverse transformation to use for displaying the images

    ``dataset_info`` is retained for compatibility with callers that pass the
    parsed dataset metadata. Loader selection and format-specific options are
    supplied by the explicit arguments below.
    """

    source = dataset_path or (dataset_info.repository if dataset_info is not None else None)
    if source is None:
        raise ValueError(
            "A dataset path is required. Set dataset_path or "
            "dataset_info.repository."
        )

    root: Path = Path(source).expanduser()
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
    # Format-specific settings are optional for direct ``get_dataloader``
    # callers.  Do not pass them to every loader: apart from making a missing
    # ``parquet_info`` crash, doing so also duplicated explicit keyword
    # arguments supplied by existing callers.
    if parquet_info is None and dataset_info is not None:
        parquet_info = dataset_info.parquet_info
    if dataset_type == "parquet" and parquet_info is not None:
        kwargs.setdefault("image_column", parquet_info.image_column)
        kwargs.setdefault("label_column", parquet_info.label_column)
        kwargs.setdefault("image_key", parquet_info.image_key)

    dataset: Dataset = loader(
        root=root,
        transform=transform if transform is not None else get_transformation(),
        split=folder_data,
        **kwargs,
    )

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
