"""App-owned privacy dataset registry and concrete dataset wrappers."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, Subset


@dataclass(frozen=True)
class PrivacyDatasetRuntimeConfig:
    dataset_id: str
    root: Path
    task_attr: str | None
    use_embeddings: bool
    max_samples: int | None
    seed: int


class _TensorClassificationDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x.float()
        self.y = y.long()

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class _ImagePathClassificationDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        labels: torch.Tensor,
        *,
        image_size: tuple[int, int] | None = None,
        channels: int = 3,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels.long()
        self.image_size = image_size
        self.channels = channels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[index])
        image = image.convert("L" if self.channels == 1 else "RGB")
        if self.image_size is not None:
            image = image.resize(self.image_size, Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        if array.ndim == 2:
            array = array[None, :, :]
        else:
            array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array), self.labels[index]


class _BasePrivacyDatasetHandle:
    config: PrivacyDatasetRuntimeConfig
    num_classes: int
    full_dataset: Dataset

    def build_subset(self, indices: list[int]) -> Subset:
        return Subset(self.full_dataset, [int(index) for index in indices])


class Cifar10PrivacyDataset(_BasePrivacyDatasetHandle):
    """CIFAR-10 wrapper backed by the local python-batches layout."""

    def __init__(self, config: PrivacyDatasetRuntimeConfig) -> None:
        self.config = config
        data_dir = config.root / "cifar10" / "raw" / "cifar-10-batches-py"
        if not data_dir.exists():
            raise FileNotFoundError(f"CIFAR-10 python batch directory not found: {data_dir}")

        arrays: list[np.ndarray] = []
        labels: list[int] = []
        for filename in [*(f"data_batch_{i}" for i in range(1, 6)), "test_batch"]:
            with open(data_dir / filename, "rb") as handle:
                batch = pickle.load(handle, encoding="latin1")
            arrays.append(batch["data"])
            labels.extend(int(label) for label in batch["labels"])

        x_np = np.concatenate(arrays, axis=0).reshape(-1, 3, 32, 32)
        x = torch.from_numpy(x_np).float() / 255.0
        y = torch.tensor(labels, dtype=torch.long)
        x, y, _ = _select_max_samples_with_indices(x, y, max_samples=config.max_samples, seed=config.seed)
        self.full_dataset = _TensorClassificationDataset(x, y)
        self.num_classes = 10


class CelebAPrivacyDataset(_BasePrivacyDatasetHandle):
    """CelebA wrapper exposing task labels and binary attribute columns."""

    EMBEDDING_DIM = 512

    def __init__(self, config: PrivacyDatasetRuntimeConfig) -> None:
        self.config = config
        celeba_dir = config.root / "celeba" / "raw" / "celeba"
        attr_path = celeba_dir / "list_attr_celeba.txt"
        image_dir = celeba_dir / "img_align_celeba"
        if not attr_path.exists():
            raise FileNotFoundError(f"CelebA attribute file not found: {attr_path}")
        if not image_dir.exists():
            raise FileNotFoundError(f"CelebA image directory not found: {image_dir}")

        names, attributes = _read_celeba_attributes(attr_path)
        task_attr = config.task_attr or "Smiling"
        if task_attr not in attributes:
            raise ValueError(f"Unknown CelebA task_attr '{task_attr}'. Available: {sorted(attributes)}")

        image_paths: list[Path] = []
        kept_rows: list[int] = []
        for row_index, filename in enumerate(names):
            image_path = image_dir / filename
            if image_path.exists():
                image_paths.append(image_path)
                kept_rows.append(row_index)

        if not image_paths:
            raise FileNotFoundError(f"No CelebA images found under {image_dir}")

        attr_matrix = torch.stack([attributes[name][kept_rows] for name in attributes], dim=1)
        attr_names = list(attributes)
        x_indices = torch.arange(len(image_paths))
        y = attributes[task_attr][kept_rows].long()

        x_indices, y, selected = _select_max_samples_with_indices(
            x_indices,
            y,
            max_samples=config.max_samples,
            seed=config.seed,
        )
        selected_indices = selected.tolist()
        self._attribute_names = attr_names
        self._attributes = attr_matrix[selected_indices]
        self._attribute_index = {name: i for i, name in enumerate(attr_names)}
        self._image_paths = [image_paths[i] for i in selected_indices]
        if config.use_embeddings:
            embeddings = self._load_or_build_embeddings(selected_indices)
            self.full_dataset = _TensorClassificationDataset(embeddings, y)
        else:
            self.full_dataset = _ImagePathClassificationDataset(self._image_paths, y, image_size=(64, 64), channels=3)
        self.num_classes = 2

    def _load_or_build_embeddings(self, selected_indices: list[int]) -> torch.Tensor:
        embedding_path = self.config.root / "celeba" / "embeddings.pt"
        if embedding_path.exists():
            payload = torch.load(embedding_path, map_location="cpu", weights_only=True)
            embeddings = payload["embeddings"] if isinstance(payload, dict) else payload
            return torch.as_tensor(embeddings, dtype=torch.float32)[selected_indices]

        rows = []
        for image_path in self._image_paths:
            image = Image.open(image_path).convert("L").resize((32, 16), Image.BILINEAR)
            rows.append(torch.from_numpy(np.asarray(image, dtype=np.float32).reshape(-1)) / 255.0)
        embeddings = torch.stack(rows).float()
        return F.normalize(embeddings, dim=1)

    def get_binary_attribute_values(
        self,
        attribute_name: str,
        *,
        indices: list[int] | tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        try:
            attr_idx = self._attribute_index[attribute_name]
        except KeyError as exc:
            raise ValueError(f"Unknown CelebA attribute '{attribute_name}'.") from exc
        values = self._attributes[:, attr_idx]
        if indices is not None:
            values = values[list(int(index) for index in indices)]
        return values.long()

    def get_property_filtered_subset(
        self,
        *,
        property_attr: str,
        target_ratio: float,
        subset_size: int,
        seed: int,
        indices: list[int] | tuple[int, ...],
    ) -> Subset:
        if subset_size <= 0:
            raise ValueError(f"subset_size must be positive, got {subset_size}.")
        pool_indices = [int(index) for index in indices]
        attr_values = self.get_binary_attribute_values(property_attr, indices=pool_indices)
        pos = [pool_indices[i] for i, value in enumerate(attr_values.tolist()) if int(value) == 1]
        neg = [pool_indices[i] for i, value in enumerate(attr_values.tolist()) if int(value) == 0]
        if not pos or not neg:
            raise ValueError(f"Property '{property_attr}' must have both positive and negative samples.")

        n_pos = int(round(subset_size * float(target_ratio)))
        n_pos = min(max(n_pos, 0), subset_size)
        n_neg = subset_size - n_pos
        gen = torch.Generator().manual_seed(int(seed))
        chosen = _sample_indices(pos, n_pos, generator=gen) + _sample_indices(neg, n_neg, generator=gen)
        order = torch.randperm(len(chosen), generator=gen).tolist()
        return Subset(self.full_dataset, [chosen[i] for i in order])


class AttFacesPrivacyDataset(_BasePrivacyDatasetHandle):
    """AT&T/ORL faces wrapper for reconstruction experiments."""

    def __init__(self, config: PrivacyDatasetRuntimeConfig) -> None:
        self.config = config
        faces_dir = config.root / "att_faces" / "raw" / "att_faces"
        if not faces_dir.exists():
            raise FileNotFoundError(f"AT&T faces directory not found: {faces_dir}")

        image_paths: list[Path] = []
        labels: list[int] = []
        subject_dirs = sorted(
            [path for path in faces_dir.iterdir() if path.is_dir() and path.name.startswith("s")],
            key=lambda path: int(path.name[1:]),
        )
        for label, subject_dir in enumerate(subject_dirs):
            for image_path in sorted(subject_dir.glob("*.pgm"), key=lambda path: int(path.stem)):
                image_paths.append(image_path)
                labels.append(label)

        if not image_paths:
            raise FileNotFoundError(f"No AT&T face images found under {faces_dir}")

        self._subject_count = len(subject_dirs)
        self._images_per_subject = len(image_paths) // max(self._subject_count, 1)
        y = torch.tensor(labels, dtype=torch.long)
        self.full_dataset = _ImagePathClassificationDataset(image_paths, y, image_size=(64, 64), channels=1)
        self.num_classes = self._subject_count

    def get_paper_train_validation_indices(self) -> tuple[list[int], list[int]]:
        train: list[int] = []
        val: list[int] = []
        for subject_idx in range(self._subject_count):
            start = subject_idx * self._images_per_subject
            train.extend(range(start, start + 7))
            val.extend(range(start + 7, start + self._images_per_subject))
        return train, val


@dataclass(frozen=True)
class PrivacyDatasetSpec:
    dataset_id: str
    builder: Callable[..., _BasePrivacyDatasetHandle]
    description: str | None = None
    num_classes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def info(self) -> dict[str, Any]:
        return {
            "id": self.dataset_id,
            "description": self.description or f"Privacy dataset '{self.dataset_id}'",
            "num_classes": self.num_classes,
            **self.metadata,
        }


class AppPrivacyDatasetFactory:
    def __init__(self) -> None:
        self._registry: dict[str, PrivacyDatasetSpec] = {}

    def register(self, spec: PrivacyDatasetSpec) -> None:
        self._registry[spec.dataset_id] = spec

    def list_specs(self) -> list[PrivacyDatasetSpec]:
        return list(self._registry.values())

    def load_dataset(
        self,
        *,
        dataset_id: str,
        root: Path,
        seed: int = 42,
        task_attr: str | None = None,
        use_embeddings: bool = True,
        max_samples: int | None = None,
        **kwargs: Any,
    ) -> _BasePrivacyDatasetHandle:
        spec = self._registry.get(dataset_id)
        if spec is None:
            raise ValueError(f"Unknown privacy dataset '{dataset_id}'. Registered: {sorted(self._registry)}")
        config = PrivacyDatasetRuntimeConfig(
            dataset_id=dataset_id,
            root=Path(root),
            task_attr=task_attr,
            use_embeddings=bool(use_embeddings),
            max_samples=max_samples,
            seed=int(seed),
        )
        return spec.builder(config=config, **kwargs)


def build_app_privacy_dataset_factory() -> AppPrivacyDatasetFactory:
    factory = AppPrivacyDatasetFactory()
    factory.register(PrivacyDatasetSpec(
        "cifar10", Cifar10PrivacyDataset,
        description="CIFAR-10 dataset wrapper for privacy attacks.",
        num_classes=10,
        metadata={"use_embeddings": False},
    ))
    factory.register(PrivacyDatasetSpec(
        "celeba", CelebAPrivacyDataset,
        description="CelebA dataset wrapper exposing binary attributes.",
        num_classes=2,
        metadata={"use_embeddings": True},
    ))
    factory.register(PrivacyDatasetSpec(
        "att_faces", AttFacesPrivacyDataset,
        description="AT&T/ORL faces dataset wrapper for reconstruction.",
        num_classes=40,
        metadata={"use_embeddings": False},
    ))
    return factory


def _select_max_samples_with_indices(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    max_samples: int | None,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if max_samples is None or max_samples >= len(y):
        indices = torch.arange(len(y))
        return x, y, indices
    gen = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(y), generator=gen)[: int(max_samples)]
    return x[indices], y[indices], indices


def _sample_indices(values: list[int], count: int, *, generator: torch.Generator) -> list[int]:
    if count <= 0:
        return []
    if count <= len(values):
        perm = torch.randperm(len(values), generator=generator)[:count]
    else:
        perm = torch.randint(0, len(values), (count,), generator=generator)
    return [values[int(i)] for i in perm.tolist()]


def _read_celeba_attributes(attr_path: Path) -> tuple[list[str], dict[str, torch.Tensor]]:
    with open(attr_path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    attr_names = lines[1].split()
    filenames: list[str] = []
    raw_values: list[list[int]] = []
    for line in lines[2:]:
        parts = line.split()
        filenames.append(parts[0])
        raw_values.append([1 if int(value) == 1 else 0 for value in parts[1:]])
    matrix = torch.tensor(raw_values, dtype=torch.long)
    return filenames, {name: matrix[:, i] for i, name in enumerate(attr_names)}


__all__ = [
    "AppPrivacyDatasetFactory",
    "AttFacesPrivacyDataset",
    "CelebAPrivacyDataset",
    "Cifar10PrivacyDataset",
    "PrivacyDatasetRuntimeConfig",
    "PrivacyDatasetSpec",
    "build_app_privacy_dataset_factory",
]
