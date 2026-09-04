from pathlib import Path
from typing import Optional

from torchvision import transforms as T
from torch.utils.data import Dataset

from utils.dataset.dataset import ImageDatasetFolder, FlatImageDataset, ParquetImageDataset
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
        **kwargs
) -> Dataset:
    root = root if root.is_file() else _resolve_dataset_root(root, split)
    return ParquetImageDataset(
        root,
        transform=transform,
        **kwargs
    )


def _load_auto(
        root: Path,
        transform: T.Compose,
        split: Optional[str] = None,
        **kwargs
) -> Dataset:
    if root.is_file() and root.suffix.lower() == ".parquet":
        return _load_parquet(root, transform, **kwargs)
    root = _resolve_dataset_root(root, split)
    if root.is_dir() and any(root.glob("*.parquet")):
        return _load_parquet(root, transform, **kwargs)
    class_dirs = [path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")]
    has_classes = bool(class_dirs) and any(
        any(file.is_file() for file in class_dir.rglob("*")) for class_dir in class_dirs
    )
    return _load_image_folder(root, transform) if has_classes else _load_flat(root, transform)
