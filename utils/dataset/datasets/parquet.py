import bisect
import io
import logging
from pathlib import Path
from typing import Iterator, Optional

import pyarrow as pa
import pyarrow.parquet as parquet
from PIL import Image as PILImage
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms as T

logger = logging.getLogger(__name__)


class ParquetImageDataset(IterableDataset):
    """Restartable, bounded-memory image stream over Parquet shards."""

    def __init__(
            self,
            root: Path,
            transform: Optional[T.Compose] = None,
            image_column: str = "image",
            label_column: Optional[str] = "label",
            image_key: Optional[str] = "bytes",
            read_batch_size: int = 64,
            limit: Optional[int] = None,
            **kwargs,
    ):
        super().__init__()
        if root is None:
            raise ValueError("The root directory is required.")
        if not root.exists():
            raise ValueError("The root directory does not exist.")
        if not root.is_dir() and not root.is_file():
            raise ValueError("The Parquet source must be a directory or a file.")
        if root.is_file() and root.suffix.lower() != ".parquet":
            raise ValueError(f"The Parquet source must end in '.parquet': {root}.")
        if read_batch_size <= 0:
            raise ValueError("read_batch_size must be greater than zero.")

        self.transform = transform
        self.image_column = image_column
        self.label_column = label_column
        self.image_key = image_key
        self.read_batch_size = read_batch_size
        candidate_paths = [root] if root.is_file() else sorted(root.glob("*.parquet"))
        if not candidate_paths:
            raise ValueError(f"No Parquet files were found in {root}.")

        self.paths: list[Path] = []
        self.row_groups: list[tuple[int, int, int, int]] = []
        invalid_schemas: list[tuple[Path, set[str]]] = []
        unreadable_files: list[tuple[Path, str]] = []
        total = 0
        for path in candidate_paths:
            try:
                parquet_file = parquet.ParquetFile(path)
            except (OSError, pa.ArrowException) as error:
                unreadable_files.append((path, str(error)))
                continue
            schema_names = set(parquet_file.schema_arrow.names)
            if image_column not in schema_names or (
                    label_column is not None and label_column not in schema_names):
                invalid_schemas.append((path, schema_names))
                continue

            file_index = len(self.paths)
            self.paths.append(path)
            for row_group_index in range(parquet_file.num_row_groups):
                row_group = parquet_file.metadata.row_group(row_group_index)
                if label_column is not None:
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
                self.row_groups.append((total, file_index, row_group_index, rows))
                total += rows

        if not self.paths:
            details = "; ".join(
                f"{path.name}: {sorted(columns)}" for path, columns in invalid_schemas[:3])
            unreadable_details = "; ".join(
                f"{path.name}: {reason}" for path, reason in unreadable_files[:3])
            available_details = "; ".join(
                detail for detail in (details, unreadable_details) if detail) or "none"
            raise ValueError(
                "No usable Parquet files match the configured image and label columns "
                f"({image_column!r}, {label_column!r}) in {root}. Details: {available_details}.")
        if unreadable_files:
            logger.warning("Skipping %d unreadable Parquet shard(s): %s",
                           len(unreadable_files),
                           ", ".join(str(path) for path, _ in unreadable_files))
        self.length = total if limit is None or limit < 0 else min(limit, total)
        self.group_starts = [group[0] for group in self.row_groups]

    def __len__(self) -> int:
        return self.length

    def _decode_image(self, image_value):
        if isinstance(image_value, dict):
            if self.image_key is None:
                raise ValueError(
                    "The Parquet image value is a struct; set image_key to the "
                    f"encoded-bytes field. Available keys: {sorted(image_value)}.")
            try:
                image_value = image_value[self.image_key]
            except KeyError:
                raise ValueError(
                    f"Image key {self.image_key!r} was not found in the Parquet "
                    f"image value. Available keys: {sorted(image_value)}.") from None
        if not isinstance(image_value, (bytes, bytearray, memoryview)):
            raise ValueError("The Parquet image column must contain encoded image bytes.")
        with PILImage.open(io.BytesIO(bytes(image_value))) as image:
            sample = image.convert("RGB")
        return self.transform(sample) if self.transform is not None else sample

    def __getitem__(self, index: int) -> tuple[object, int]:
        """Compatibility path for callers that inspect an individual sample.

        DataLoader recognizes this class as an IterableDataset and therefore
        uses ``__iter__`` below. Direct indexing still performs a bounded
        record-batch scan rather than materializing a complete row group.
        """
        if index < 0 or index >= len(self):
            raise IndexError(index)
        group_index = bisect.bisect_right(self.group_starts, index) - 1
        start, file_index, row_group_index, _ = self.row_groups[group_index]
        offset = index - start
        columns = [self.image_column]
        if self.label_column is not None:
            columns.append(self.label_column)
        parquet_file = parquet.ParquetFile(self.paths[file_index])
        try:
            seen = 0
            for record_batch in parquet_file.iter_batches(
                    batch_size=self.read_batch_size, row_groups=[row_group_index],
                    columns=columns, use_threads=False):
                if offset >= seen + record_batch.num_rows:
                    seen += record_batch.num_rows
                    continue
                row_index = offset - seen
                image_value = record_batch.column(self.image_column)[row_index].as_py()
                target = (-1 if self.label_column is None else
                          int(record_batch.column(self.label_column)[row_index].as_py()))
                return self._decode_image(image_value), target
        finally:
            parquet_file.close()
        raise IndexError(index)

    def __iter__(self) -> Iterator[tuple[object, int]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        columns = [self.image_column]
        if self.label_column is not None:
            columns.append(self.label_column)
        open_files: dict[int, parquet.ParquetFile] = {}
        try:
            for group_number, (start, file_index, row_group_index, rows) in enumerate(self.row_groups):
                if start >= self.length:
                    break
                if group_number % num_workers != worker_id:
                    continue
                parquet_file = open_files.get(file_index)
                if parquet_file is None:
                    parquet_file = parquet.ParquetFile(self.paths[file_index])
                    open_files[file_index] = parquet_file
                remaining = min(rows, self.length - start)
                emitted = 0
                for record_batch in parquet_file.iter_batches(
                        batch_size=self.read_batch_size, row_groups=[row_group_index],
                        columns=columns, use_threads=False):
                    take = min(record_batch.num_rows, remaining - emitted)
                    image_values = record_batch.column(self.image_column)
                    label_values = (record_batch.column(self.label_column)
                                    if self.label_column is not None else None)
                    for row_index in range(take):
                        image_value = image_values[row_index].as_py()
                        target = (-1 if label_values is None
                                  else int(label_values[row_index].as_py()))
                        yield self._decode_image(image_value), target
                    emitted += take
                    if emitted >= remaining:
                        break
        finally:
            for parquet_file in open_files.values():
                parquet_file.close()
