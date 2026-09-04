import bisect
import io
from pathlib import Path
from typing import Mapping, Optional

import torchvision
from PIL import Image as PILImage
from torch.utils.data import Dataset


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


class FlatImageDataset(Dataset):
    """Images in one directory, optionally accompanied by a labels file.

    A labels file can be either ``labels.csv`` (columns ``file`` and ``label``)
    or ``labels.json`` (a mapping from file name to integer label).  Keeping
    this adapter separate from ImageFolder is important: a directory such as
    ``images/`` is not a class called images.
    """

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

    def __init__(self, root: Path, transform=None, labels: Optional[Mapping[str, int]] = None):
        self.root = root
        self.transform = transform
        self.paths = sorted(
            path for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in self.extensions
        )
        self.labels = labels
        if not self.paths:
            raise ValueError(f"No supported images were found in {root}.")
        if labels is not None:
            missing = [path.name for path in self.paths if path.name not in labels]
            if missing:
                raise ValueError(f"The labels file has no label for {missing[0]!r}.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with PILImage.open(self.paths[index]) as image:
            sample = image.convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        # -1 makes an unlabeled image dataset usable for inference while
        # making the absence of labels visible to classification metrics.
        target = -1 if self.labels is None else int(self.labels[self.paths[index].name])
        return sample, target


class ParquetImageDataset(Dataset):
    """Images stored in one or more Parquet files."""

    def __init__(
            self,
            root: Path,
            transform=None,
            image_column: str = "image",
            label_column: Optional[str] = "label",
    ):
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise ImportError(
                "Parquet datasets require the 'pyarrow' package."
            ) from error

        self.root = root
        self.transform = transform
        self.paths = [root] if root.is_file() else sorted(root.glob("*.parquet"))
        if not self.paths:
            raise ValueError(f"No Parquet files were found in {root}.")

        self.files = [parquet.ParquetFile(path) for path in self.paths]
        schema_names = set(self.files[0].schema_arrow.names)
        if image_column not in schema_names:
            raise ValueError(
                f"Image column {image_column!r} was not found in {self.paths[0]}. "
                f"Available columns: {sorted(schema_names)}."
            )
        if label_column is not None and label_column not in schema_names:
            raise ValueError(
                f"Label column {label_column!r} was not found in {self.paths[0]}. "
                f"Available columns: {sorted(schema_names)}."
            )
        self.image_column = image_column
        self.label_column = label_column
        self.row_groups = []
        total = 0
        for file_index, parquet_file in enumerate(self.files):
            for row_group_index in range(parquet_file.num_row_groups):
                rows = parquet_file.metadata.row_group(row_group_index).num_rows
                self.row_groups.append((total, file_index, row_group_index))
                total += rows
        self.group_starts = [group[0] for group in self.row_groups]
        self.length = total
        self._cached_group = None
        self._cached_rows = None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        group_index = bisect.bisect_right(self.group_starts, index) - 1
        start, file_index, row_group_index = self.row_groups[group_index]
        cache_key = (file_index, row_group_index)
        if self._cached_group != cache_key:
            columns = [self.image_column]
            if self.label_column is not None:
                columns.append(self.label_column)
            self._cached_rows = self.files[file_index].read_row_group(
                row_group_index, columns=columns
            ).to_pylist()
            self._cached_group = cache_key

        row = self._cached_rows[index - start]
        image_value = row[self.image_column]
        if isinstance(image_value, dict):
            image_value = image_value.get("bytes")
        if not isinstance(image_value, (bytes, bytearray, memoryview)):
            raise ValueError("The Parquet image column must contain encoded image bytes.")
        with PILImage.open(io.BytesIO(bytes(image_value))) as image:
            sample = image.convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        target = -1 if self.label_column is None else int(row[self.label_column])
        return sample, target
