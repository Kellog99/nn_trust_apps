import json
from io import BytesIO
from zipfile import ZipFile

import pytest
from PIL import Image

from models.info import DatasetInfo
from utils.model import download_yolo


def test_prepare_dataset_extracts_coco_validation_and_metadata(tmp_path, monkeypatch):
    image = BytesIO()
    Image.new("RGB", (8, 8)).save(image, format="JPEG")

    annotations_archive = tmp_path / "annotations.zip"
    with ZipFile(annotations_archive, "w") as archive:
        archive.writestr("annotations/instances_val2017.json", json.dumps({
            "images": [{"id": 1, "file_name": "000000000001.jpg"}],
            "annotations": [],
            "categories": [{"id": 1, "name": "person"}],
        }))
    images_archive = tmp_path / "images.zip"
    with ZipFile(images_archive, "w") as archive:
        archive.writestr("val2017/000000000001.jpg", image.getvalue())

    def local_archive(url, destination):
        return annotations_archive if "annotations" in url else images_archive

    monkeypatch.setattr(download_yolo, "download_archive", local_archive)
    dataset_dir = tmp_path / "dataset"
    download_yolo.prepare_dataset(dataset_dir)

    info = DatasetInfo.model_validate_json((dataset_dir / "info.json").read_text())
    assert info.dataset_type == "coco"
    assert info.num_samples == 1
    assert info.repository == str(dataset_dir)
    assert (dataset_dir / info.annotations_file).is_file()
    assert (dataset_dir / info.images_dir / "000000000001.jpg").is_file()

    # An existing dataset should not download either archive again.
    monkeypatch.setattr(download_yolo, "download_archive", lambda *_: pytest.fail("downloaded again"))
    download_yolo.prepare_dataset(dataset_dir)
