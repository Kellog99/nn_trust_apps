import bisect
import io
from pathlib import Path
from typing import Optional

import pyarrow.parquet as parquet
from PIL import Image as PILImage
from torch.utils.data import Dataset


class ParquetImageDataset(Dataset):
    """
    Images stored in one or more Parquet files.
    """

    def __init__(
            self,
            root: Path,
            transform=None,
            image_column: str = "image",
            label_column: Optional[str] = "label",
            image_key: str = "bytes",
            **kwargs
    ):

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
        self.image_key = image_key
        self.row_groups = []
        total = 0
        for file_index, parquet_file in enumerate(self.files):
            for row_group_index in range(parquet_file.num_row_groups):
                rows = parquet_file.metadata.row_group(row_group_index).num_rows
                self.row_groups.append((total, file_index, row_group_index))
                total += rows
        self.group_starts = [group[0] for group in self.row_groups]
        self.length = total

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self):
            raise IndexError(index)

        # Extracting the index of the group that contains the index
        group_index = bisect.bisect_right(self.group_starts, index) - 1
        start, file_index, row_group_index = self.row_groups[group_index]

        rows = self.files[file_index].read_row_group(row_group_index).to_pylist()

        row = rows[index - start]
        image_value = row[self.image_column]

        if isinstance(image_value, dict):
            image_value = image_value.get(self.image_key)
        if not isinstance(image_value, (bytes, bytearray, memoryview)):
            raise ValueError("The Parquet image column must contain encoded image bytes.")
        with PILImage.open(io.BytesIO(bytes(image_value))) as image:
            sample = image.convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        target = -1 if self.label_column is None else int(row[self.label_column])
        return sample, target
