from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset
from torchvision import transforms as T

from utils.dataset.datasets import ImageDatasetFolder, FlatImageDataset, ParquetImageDataset
from utils.dataset.utils import _read_labels, _resolve_dataset_root, _is_valid_image


def _load_image_folder(
        root: Path,
        transform: T.Compose,
        split: Optional[str] = None,
        **kwargs
) -> Dataset:
    root = _resolve_dataset_root(root, split)
    return ImageDatasetFolder(
        str(root),
        transform=transform,
        is_valid_file=lambda filename: _is_valid_image(filename),
    )


def _load_flat(
        root: Path,
        transform: T.Compose,
        split: Optional[str] = None,
        **kwargs
) -> Dataset:
    root = _resolve_dataset_root(root, split)
    return FlatImageDataset(
        root,
        transform=transform,
        labels=_read_labels(root)
    )


def _load_parquet(
        root: Path,
        transform: T.Compose,
        split: Optional[str] = None,
        image_column: str = "image",
        label_column: str = "label",
        image_key: str = "image",
        **kwargs
) -> Dataset:
    root = root if root.is_file() else _resolve_dataset_root(root, split)
    return ParquetImageDataset(
        root,
        transform=transform,
        image_column=image_column,
        label_column=label_column,
        image_key=image_key,
        **kwargs
    )
