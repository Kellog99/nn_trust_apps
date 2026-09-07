from pathlib import Path
from typing import get_args

import pytest
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from models.info import DATASET_TYPES
from utils.load_dataset import _LOADERS


DATASET_ROOT = Path("~/Desktop/StableAI/dataset_repository").expanduser()

# Exercise every loader type against a real layout in the local repository.
ROOT_DIR: dict[DATASET_TYPES, tuple[Path, str | None]] = {
    "auto": (DATASET_ROOT / "imagenette2", "train"),
    "image_folder": (DATASET_ROOT / "imagenette2", "val"),
    "flat": (DATASET_ROOT / "imagenette2" / "train" / "n01440764", None),
    "parquet": (
        DATASET_ROOT / "imagenet-1k" / "data" / "test-00000-of-00028.parquet",
        None,
    ),
}


@pytest.mark.parametrize("dataset_type", get_args(DATASET_TYPES))
def test_dataset_loader(dataset_type: DATASET_TYPES):
    path, split = ROOT_DIR[dataset_type]
    assert path.exists(), f"Dataset path does not exist: {path}"
    assert path.is_dir() or path.suffix.lower() == ".parquet"

    dataset: Dataset = _LOADERS[dataset_type](
        root=path,
        transform=ToTensor(),
        split=split,
    )

    assert len(dataset) > 0
    sample, label = dataset[0]
    assert isinstance(sample, Tensor)
    assert sample.ndim == 3
    assert isinstance(label, int)
