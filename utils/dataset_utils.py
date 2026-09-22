import os
import random
from pathlib import Path
from typing import Callable, Optional

import numpy
import torch
from torch.utils.data import Dataset, IterableDataset, Subset, DataLoader
from torchvision.transforms import transforms
from torchvision.transforms import v2

from models.info import DATASET_TYPES, ParquetInfo, Transformation
from nn_trust import Task
from utils.dataset._load_classification_dataset import _load_image_folder, _load_flat, _load_parquet
from utils.dataset._load_od_dataset import _load_coco
from utils.dataset.datasets.coco import CocoDetectionDataset
from utils.dataset.datasets.folder import ImageDatasetFolder


_LOADERS: dict[DATASET_TYPES, Callable[..., Dataset]] = {
    "coco": _load_coco,
    "image_folder": _load_image_folder,
    "flat": _load_flat,
    "parquet": _load_parquet,
}


def get_transform_classification(transformation: Transformation | None) -> transforms.Compose:
    """Convert a PIL image or NumPy array to a classification tensor.

    With a configuration, optionally resize to a square of ``size`` pixels,
    center-crop to ``crop``, then normalize using the supplied mean and std.
    Normalizing last ensures any crop padding represents black pixels.
    With None, only convert to a tensor (uint8 pixels are scaled to [0, 1]).
    """
    out = [transforms.ToTensor()]
    if transformation is None:
        return transforms.Compose(out)
    if transformation.size is not None:
        out.append(transforms.Resize((transformation.size, transformation.size)))
    if transformation.crop is not None:
        out.append(transforms.CenterCrop(transformation.crop))
    out.append(transforms.Normalize(mean=transformation.mean, std=transformation.std))
    return transforms.Compose(out)


def get_inverse_transform(
        transformation: Transformation | transforms.Compose | transforms.Normalize | v2.Compose | v2.Transform | None,
        H: int,
        W: int
) -> transforms.Compose:
    """Build a display inverse for classification preprocessing.

    Args:
        transformation: Model configuration, a torchvision v1/v2 pipeline or
            Normalize, or None for an unnormalized image. Nested Compose objects
            are supported; normalizations are undone in reverse order.
        H: Original image height (positive).
        W: Original image width (positive).

    Returns:
        A Compose accepting floating tensors shaped (C, H, W) or (N, C, H, W),
        preserving their device and dtype. Channel counts follow the supplied
        normalization statistics, including grayscale. Values are not clamped,
        and tensor conversion is not reversed into PIL or integer pixels.

        Resize and CenterCrop are approximated by resizing to (H, W) after
        undoing normalization: lost pixels and crop coordinates are not restored.
        ToTensor and scaled floating-point dtype conversions require no inverse
        for display. None performs only the final resize.

    Raises:
        ValueError: Dimensions or normalization statistics are invalid.
        TypeError: A transform is unsupported (e.g. random augmentation, Lambda,
            or unscaled dtype conversion). Its inverse needs additional state.
    """
    if H <= 0 or W <= 0:
        raise ValueError("H and W must be positive.")

    inverse = []
    pending = [transformation]
    while pending:
        transform = pending.pop()
        if transform is None or isinstance(transform, torch.nn.Identity):
            continue
        elif isinstance(transform, (transforms.Compose, v2.Compose)):
            pending.extend(transform.transforms)
        elif isinstance(transform, (Transformation, transforms.Normalize, v2.Normalize)):
            mean = numpy.asarray(transform.mean, dtype=float).reshape(-1)
            std = numpy.asarray(transform.std, dtype=float).reshape(-1)
            if (not mean.size or mean.size != std.size
                    or not numpy.isfinite(mean).all() or not numpy.isfinite(std).all()
                    or (std <= 0).any()):
                raise ValueError("mean and std must be finite, equally sized, nonempty; std must be positive.")
            inverse.append(transforms.Normalize((-mean / std).tolist(), (1 / std).tolist()))
        elif isinstance(transform, (transforms.ToTensor, v2.ToTensor,
                                    transforms.Resize, v2.Resize,
                                    transforms.CenterCrop, v2.CenterCrop, v2.ToImage)):
            continue
        elif isinstance(transform, (transforms.ConvertImageDtype, v2.ConvertImageDtype)):
            if not transform.dtype.is_floating_point:
                raise TypeError("Only floating-point dtype conversions are supported.")
        elif isinstance(transform, v2.ToDtype):
            if not (isinstance(transform.dtype, torch.dtype)
                    and transform.dtype.is_floating_point and transform.scale):
                raise TypeError("ToDtype must use a floating-point dtype and scale=True.")
        else:
            raise TypeError(f"Cannot invert {type(transform).__name__} without additional state.")

    return transforms.Compose([*inverse, transforms.Resize((H, W))])


def get_dataloader(
        dataset_path: str,
        batch: int,
        model_transformation: Transformation | None,
        subset: Optional[int] = None,
        num_workers: int = 4,
        name: Optional[str] = None,
        model_type: Optional[str] = None,
        task: Optional[Task] = None,
        images_dir: Optional[str] = None,
        annotations_file: Optional[str] = None,
        dataset_type: DATASET_TYPES = "image_folder",
        folder_data: Optional[str] = None,
        parquet_info: Optional[ParquetInfo] = None,
        **kwargs,
) -> DataLoader:
    """
    Load the format selected by DatasetInfo.dataset_type and return its DataLoader.
    """
    dataset_path = Path(dataset_path).expanduser()

    if not os.path.exists(dataset_path):
        raise ValueError(f"The dataset --------{dataset_path} does not exists.")

    try:
        loader = _LOADERS[dataset_type]
    except KeyError:
        raise ValueError(
            f"Unsupported dataset type: {dataset_type}. Supported types: {sorted(_LOADERS)}."
        ) from None

    if task is not None:
        expected_task = Task.Detection if dataset_type == "coco" else Task.Classification
        if task != expected_task:
            raise ValueError(f"Dataset type {dataset_type!r} does not support task {task}.")

    if dataset_type == "parquet":
        if parquet_info is not None:
            kwargs.setdefault("image_column", parquet_info.image_column)
            kwargs.setdefault("label_column", parquet_info.label_column)
            kwargs.setdefault("image_key", parquet_info.image_key)
        kwargs.setdefault("read_batch_size", max(1, batch))
        kwargs.setdefault("limit", subset)

    dataset = loader(
        root=dataset_path,
        transform=get_transform_classification(model_transformation),
        split=folder_data,
        model_type=model_type,
        images_dir=images_dir,
        annotations_file=annotations_file,
        **kwargs,
    )

    dataset.name = name if name is not None else dataset_path.name

    if isinstance(dataset, IterableDataset):
        subdataset = dataset
    else:
        indices = range(len(dataset) if subset is None or subset < 0 else min(subset, len(dataset)))
        subdataset = Subset(dataset, indices)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(1234)

    dataloader_kwargs = {
        "batch_size": batch,
        "shuffle": not isinstance(dataset, IterableDataset),
        "num_workers": num_workers,
        "worker_init_fn": seed_worker,
        "generator": g,
        "pin_memory": True,
    }

    if dataset_type == "coco":
        dataloader_kwargs["collate_fn"] = lambda batch: tuple(zip(*batch))

    dataloader = DataLoader(subdataset, **dataloader_kwargs)

    return dataloader
