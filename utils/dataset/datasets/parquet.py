import bisect
import io
import logging
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as parquet
from PIL import Image as PILImage
from torch.utils.data import Dataset
from torchvision import transforms as T


logger = logging.getLogger(__name__)


class ParquetImageDataset(Dataset):
    """
    Images stored in one or more Parquet files.
    """

    def __init__(
            self,
            root: Path,
            transform: Optional[T.Compose] = None,
            image_column: str = "image",
            label_column: Optional[str] = "label",
            image_key: Optional[str] = "bytes",
            **kwargs
    ):
        """
        This represents the class for the parquet image dataset

        :param root: it is the directory to the (.parquet) files
        :param transform:  the transformation to use on the data
        :param image_column: It is the column where the data is
        :param label_column: it is the column where the label is
        :param image_key: if the entry in the image column is a dict, this takes the key
        :param kwargs:
        """

        # Checking the condition of the root path
        if root is None:
            raise ValueError("The root directory is required.")
        elif not root.exists():
            raise ValueError("The root directory does not exist.")
        elif not root.is_dir() and not root.is_file():
            raise ValueError("The Parquet source must be a directory or a file.")
        elif root.is_file() and root.suffix.lower() != ".parquet":
            raise ValueError(f"The Parquet source must end in '.parquet': {root}.")

        # transformation
        self.transform = transform

        # List all the data in the root that are parquet
        candidate_paths = [root] if root.is_file() else sorted(root.glob("*.parquet"))
        if not candidate_paths:
            raise ValueError(f"No Parquet files were found in {root}.")

        # A dataset directory can contain unrelated Parquet exports.  Select
        # the shards with the configured schema instead of failing because an
        # adjacent file belongs to another dataset.  A single supplied file is
        # still validated strictly, so configuration mistakes remain clear.
        self.paths: list[Path] = []
        self.files: list[parquet.ParquetFile] = []
        invalid_schemas: list[tuple[Path, set[str]]] = []
        unreadable_files: list[tuple[Path, str]] = []
        for path in candidate_paths:
            try:
                parquet_file = parquet.ParquetFile(path)
            except (OSError, pa.ArrowException) as error:
                # A partially copied shard, or a non-Parquet file with a
                # ``.parquet`` extension, should not make an otherwise valid
                # dataset unusable.  Keep scanning the other shards.
                unreadable_files.append((path, str(error)))
                continue
            schema_names = set(parquet_file.schema_arrow.names)
            has_columns = image_column in schema_names and (
                label_column is None or label_column in schema_names
            )
            if has_columns:
                self.paths.append(path)
                self.files.append(parquet_file)
            else:
                invalid_schemas.append((path, schema_names))
        if not self.paths:
            details = "; ".join(
                f"{path.name}: {sorted(columns)}"
                for path, columns in invalid_schemas[:3]
            )
            unreadable_details = "; ".join(
                f"{path.name}: {reason}"
                for path, reason in unreadable_files[:3]
            )
            available_details = "; ".join(
                detail for detail in (details, unreadable_details) if detail
            ) or "none"
            raise ValueError(
                "No usable Parquet files match the configured image and label columns "
                f"({image_column!r}, {label_column!r}) in {root}. "
                f"Details: {available_details}."
            )
        if unreadable_files:
            skipped_paths = ", ".join(str(path) for path, _ in unreadable_files)
            logger.warning(
                "Skipping %d unreadable Parquet shard(s); training will use "
                "the remaining valid shards: %s",
                len(unreadable_files),
                skipped_paths,
            )
        self.image_column = image_column
        self.label_column = label_column
        self.image_key = image_key
        self.row_groups = []
        total = 0
        for file_index, parquet_file in enumerate(self.files):
            for row_group_index in range(parquet_file.num_row_groups):
                row_group = parquet_file.metadata.row_group(row_group_index)
                # ImageNet test shards in the supplied repository use ``-1``
                # as an unlabeled sentinel.  Classification metrics cannot be
                # evaluated on those rows.  Skip a whole group only when its
                # parquet statistics prove that every label is negative; groups
                # without statistics are retained and validated normally.
                if label_column is not None:
                    # ``schema_arrow`` indexes a struct image as one field,
                    # while row-group metadata indexes its physical children.
                    column_index = parquet_file.schema.names.index(label_column)
                    statistics = row_group.column(column_index).statistics
                    if statistics is not None and statistics.has_min_max:
                        try:
                            is_unlabeled_group = int(statistics.max) < 0
                        except (TypeError, ValueError, OverflowError):
                            is_unlabeled_group = False
                        if is_unlabeled_group:
                            continue
                rows = row_group.num_rows
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

        columns = [self.image_column]
        if self.label_column is not None:
            columns.append(self.label_column)
        rows = self.files[file_index].read_row_group(
            row_group_index, columns=columns
        ).to_pylist()

        row = rows[index - start]
        image_value = row[self.image_column]

        if isinstance(image_value, dict):
            if self.image_key is None:
                raise ValueError(
                    "The Parquet image value is a struct; set image_key to the "
                    f"encoded-bytes field. Available keys: {sorted(image_value)}."
                )
            try:
                image_value = image_value[self.image_key]
            except KeyError:
                raise ValueError(
                    f"Image key {self.image_key!r} was not found in the Parquet "
                    f"image value. Available keys: {sorted(image_value)}."
                ) from None
        if not isinstance(image_value, (bytes, bytearray, memoryview)):
            raise ValueError("The Parquet image column must contain encoded image bytes.")
        with PILImage.open(io.BytesIO(bytes(image_value))) as image:
            sample = image.convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        target: int = -1 if self.label_column is None else int(row[self.label_column])
        return sample, target
