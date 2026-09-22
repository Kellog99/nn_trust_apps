import random
from pathlib import Path
from typing import Callable, Optional

import numpy
import torch
from torch.utils.data import Dataset, IterableDataset, Subset, DataLoader
from torchvision import transforms as T

from models.info import DATASET_TYPES, DatasetInfo, ParquetInfo
from nn_trust import Task
from utils.dataset_utils import get_transform_dataset
from utils.dataset._load_od_dataset import _load_coco
from utils.dataset._load_classification_dataset import (
    _load_image_folder,
    _load_flat,
    _load_parquet
)

_LOADERS: dict[DATASET_TYPES, Callable[..., Dataset]] = {
    "coco": _load_coco,
    "image_folder": _load_image_folder,
    "flat": _load_flat,
    "parquet": _load_parquet,
}


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def _collate_detection(batch):
    return tuple(zip(*batch))


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
        task: Optional[Task] = None,
        images_dir: Optional[str] = None,
        annotations_file: Optional[str] = None,
        **kwargs,
) -> DataLoader:
    """
    Return an ordered DataLoader for the explicitly selected dataset format.

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
            "For YOLO, video, medical volumes, or another custom "
            "format, pass a torch.utils.data.Dataset instance."
        ) from None
    if task is not None:
        expected_task = Task.Detection if dataset_type == "coco" else Task.Classification
        if task != expected_task:
            raise ValueError(f"Dataset type {dataset_type!r} does not support task {task}.")

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

    if dataset_type == "parquet":
        # Arrow decodes only a consumer-sized chunk at once. The limit is
        # applied inside the stream, without allocating a list of row indexes.
        kwargs.setdefault("read_batch_size", max(1, batch))
        kwargs.setdefault("limit", subset)

    if dataset_type == "coco":
        kwargs.update(images_dir=images_dir, annotations_file=annotations_file)

    dataset: Dataset = loader(
        root=root,
        transform=transform if transform is not None else get_transform_dataset(),
        split=folder_data,
        **kwargs,
    )

    if isinstance(dataset, IterableDataset):
        subdataset = dataset
    else:
        if subset is None or subset < 0:
            indices = range(len(dataset))
        else:
            indices = range(min(subset, len(dataset)))
        subdataset = Subset(dataset, indices)

    dataloader = DataLoader(
        subdataset,
        batch_size=batch,
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(1234),
        collate_fn=_collate_detection if dataset_type == "coco" else None,
    )
    return dataloader
