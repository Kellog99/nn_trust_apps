import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision import transforms as T

from utils.load_dataset import get_dataloader


def _png(value: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), (value, value, value)).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_shard(path: Path, labels: list[int]) -> None:
    table = pa.table({
        "image": [_png(label) for label in labels],
        "label": labels,
    })
    pq.write_table(table, path, row_group_size=2)


def test_parquet_loader_is_restartable_bounded_stream(tmp_path: Path) -> None:
    _write_shard(tmp_path / "part-0.parquet", [0, 1, 2, 3, 4])
    dataloader = get_dataloader(
        dataset_type="parquet",
        dataset_path=tmp_path,
        batch=2,
        subset=4,
        num_workers=0,
        transform=T.ToTensor(),
        image_key=None,
    )

    assert isinstance(dataloader.dataset, IterableDataset)
    assert dataloader.dataset.read_batch_size == 2
    assert len(dataloader.dataset) == 4
    first_pass = [label.item() for _, labels in dataloader for label in labels]
    second_pass = [label.item() for _, labels in dataloader for label in labels]
    assert first_pass == second_pass == [0, 1, 2, 3]

    sample, label = dataloader.dataset[0]
    assert sample.shape == (3, 2, 2)
    assert label == 0


def test_parquet_workers_partition_row_groups_without_duplicates(tmp_path: Path) -> None:
    _write_shard(tmp_path / "part-0.parquet", list(range(8)))
    dataloader = get_dataloader(
        dataset_type="parquet",
        dataset_path=tmp_path,
        batch=2,
        num_workers=2,
        transform=T.ToTensor(),
        image_key=None,
    )
    labels = [label.item() for _, batch_labels in dataloader for label in batch_labels]
    assert sorted(labels) == list(range(8))
