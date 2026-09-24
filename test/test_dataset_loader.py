import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision import transforms as T

from models.info import DatasetInfo, ParquetInfo, Transformation
from nn_trust import Task
from utils.dataset.datasets.coco import CocoDetectionDataset
from utils.dataset_utils import get_dataloader as legacy_get_dataloader, get_inverse_transform, get_transform_dataset
from utils.load_dataset import get_dataloader


def test_merge_preserves_primary_loader_behavior(tmp_path):
    """Metadata paths, explicit transforms and ordered subsets retain priority."""
    for index in range(3):
        Image.new("RGB", (4, 4), (index * 50,) * 3).save(tmp_path / f"{index}.png")
    info = DatasetInfo.model_construct(repository=str(tmp_path), parquet_info=None)
    loader = get_dataloader(
        None, 2, info, T.ToTensor(), subset=2, num_workers=-1,
        dataset_type="flat",
    )
    images, _ = next(iter(loader))
    assert loader.num_workers == 0
    assert len(loader.dataset) == 2
    assert images.shape == (2, 3, 4, 4)
    torch.testing.assert_close(images[:, 0, 0, 0], torch.tensor([0., 50 / 255]))


def test_legacy_loader_delegates_model_transformation(tmp_path):
    Image.new("RGB", (4, 4), (255,) * 3).save(tmp_path / "image.png")
    loader = legacy_get_dataloader(
        str(tmp_path), 1, Transformation(mean=[0.5] * 3, std=[0.5] * 3, size=8),
        dataset_type="flat", num_workers=0,
    )
    images, _ = next(iter(loader))
    torch.testing.assert_close(images, torch.ones(1, 3, 8, 8))


def test_incompatible_task_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="does not support task"):
        get_dataloader(tmp_path, 1, dataset_type="flat", task=Task.Detection)


@pytest.mark.parametrize("dataset_type", ["image_folder", "flat"])
def test_classification_formats_use_split_and_subset(tmp_path, dataset_type):
    """Load the requested image format and split, applying the subset."""
    root = tmp_path / "validation_images"
    image_root = root / "class_a" if dataset_type == "image_folder" else root
    image_root.mkdir(parents=True)
    for index in range(3):
        Image.new("RGB", (4, 4)).save(image_root / f"{index}.png")
    loader = get_dataloader(
        str(tmp_path), 2, None, dataset_type=dataset_type,
        folder_data="validation_images", subset=2, num_workers=0,
        task=Task.Classification,
    )
    images, labels = next(iter(loader))
    assert images.shape == (2, 3, 4, 4)
    assert len(labels) == len(loader.dataset) == 2


def test_parquet_metadata_and_streaming_subset(tmp_path):
    """Decode configured Parquet columns and limit the streamed samples."""
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="PNG")
    pq.write_table(pa.table({
        "pixels": [{"encoded": buffer.getvalue()}] * 3,
        "target": [1, 2, 3],
    }), tmp_path / "data.parquet")
    loader = get_dataloader(
        str(tmp_path), 2, None, dataset_type="parquet", subset=2, num_workers=0,
        parquet_info=ParquetInfo(image_column="pixels", image_key="encoded", label_column="target"),
    )
    assert isinstance(loader.dataset, IterableDataset)
    assert [label.item() for _, labels in loader for label in labels] == [1, 2]


def test_coco_custom_paths_and_ragged_batches(tmp_path: Path):
    """Load custom COCO paths, map categories, and batch unequal target counts."""
    images = tmp_path / "pictures"
    images.mkdir()
    for index in (1, 2):
        Image.new("RGB", (16, 16)).save(images / f"{index}.png")
    annotations = {
        "images": [
            {
                "id": index,
                "file_name": f"{index}.png",
                "height": 16,
                "width": 16
            }
            for index in (1, 2)
        ],
        "categories": [
            {
                "id": 7,
                "name":
                    "object"
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 7,
                "bbox": [1, 1, 4, 4],
                "area": 16,
                "iscrowd": 0
            }
        ],
    }
    (tmp_path / "labels.json").write_text(json.dumps(annotations))
    loader = get_dataloader(
        str(tmp_path),
        batch=2,
        transform=None,
        dataset_type="coco",
        task=Task.Detection,
        images_dir="pictures",
        annotations_file="labels.json",
        new_shape=(32, 32),
        num_workers=0,
    )
    assert isinstance(loader.dataset.dataset, CocoDetectionDataset)
    assert loader.dataset.dataset.cat_id_to_label == {7: 0}
    images, targets = next(iter(loader))
    assert len(images) == len(targets) == 2
    assert sorted(len(target["boxes"]) for target in targets) == [0, 1]
    assert all(image.shape[-2:] == (32, 32) for image in images)


def test_legacy_coco_metadata_selects_loader_and_paths(tmp_path: Path):
    """Older COCO info.json files still work through the benchmark loader path."""
    image_dir = tmp_path / "pictures"
    image_dir.mkdir()
    Image.new("RGB", (16, 16)).save(image_dir / "1.png")
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()
    (annotations_dir / "labels.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "1.png", "height": 16, "width": 16}],
        "categories": [{"id": 7, "name": "object"}],
        "annotations": [],
    }))
    info = DatasetInfo(
        id="legacy-coco", name="Legacy COCO", task="detection",
        input_dimensionality=[3, 16, 16], repository=str(tmp_path),
        images_dir="pictures", annotations_file="annotations/labels.json",
    )

    assert info.dataset_type == "coco"
    loader = get_dataloader(
        dataset_path=info.repository, batch=1, dataset_info=info,
        dataset_type=info.dataset_type, num_workers=0,
    )
    assert isinstance(loader.dataset.dataset, CocoDetectionDataset)
    images, targets = next(iter(loader))
    assert len(images) == len(targets) == 1
    assert targets[0]["boxes"].numel() == 0


def test_explicit_dataset_type_overrides_coco_inference():
    info = DatasetInfo(
        id="explicit", name="Explicit", task="detection",
        input_dimensionality=[3, 16, 16], dataset_type="image_folder",
        images_dir="pictures", annotations_file="annotations/labels.json",
    )
    assert info.dataset_type == "image_folder"


def test_unknown_dataset_type_is_rejected(tmp_path):
    """Reject formats that have no registered dataset loader."""
    with pytest.raises(ValueError, match="Unsupported dataset type"):
        get_dataloader(str(tmp_path), 1, None, dataset_type="unknown")


def test_configuration_and_generated_pipeline():
    """Restore a constant image using either its configuration or its pipeline."""
    config = Transformation(mean=[0.5] * 3, std=[0.25] * 3, size=6, crop=4)
    image = torch.full((3, 8, 10), 0.75)
    pipeline = get_transform_dataset(config)
    normalized = pipeline(T.ToPILImage()(image))
    expected = torch.full_like(image, 191 / 255)
    for source in (config, pipeline):
        torch.testing.assert_close(get_inverse_transform(source, 8, 10)(normalized), expected)


def test_classification_without_configuration():
    """Convert to a tensor without normalization when no configuration is given."""
    image = T.ToPILImage()(torch.ones(3, 4, 6))
    torch.testing.assert_close(get_transform_dataset(None)(image), torch.ones(3, 4, 6))


def test_classification_crop_padding_is_normalized_black():
    """Normalize black crop padding with the same statistics as the image."""
    config = Transformation(mean=[0.5], std=[0.25], crop=4)
    image = T.ToPILImage()(torch.ones(1, 2, 2))
    result = get_transform_dataset(config)(image)
    expected = torch.full((1, 4, 4), -2.0)
    expected[:, 1:3, 1:3] = 2.0
    torch.testing.assert_close(result, expected)
