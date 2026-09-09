import json
from pathlib import Path

import pytest
import torchvision.transforms as T
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from models.info import DATASET_TYPES, DatasetInfo
from utils.load_dataset import get_dataloader

DATASET_ROOT = Path("~/Desktop/StableAI/dataset_repository").expanduser()

# Exercise every loader type against a real layout in the local repository.
ROOT_DIR: dict[DATASET_TYPES, Path | str] = {
    "image_folder": DATASET_ROOT / "imagenette2",
    "parquet": DATASET_ROOT / "imagenet-1k",
}


@pytest.mark.parametrize("dataset_type", ROOT_DIR.keys())
def test_dataset_loader(dataset_type: DATASET_TYPES):
    path: Path | str = ROOT_DIR[dataset_type]
    if isinstance(path, str):
        path: Path = Path(path).expanduser().resolve()

    assert path.exists(), f"Dataset path does not exist: {path}"
    assert path.is_dir() or path.suffix.lower() == ".parquet"

    with open(path / "info.json", "r") as f:
        dataset_info = json.load(f)

    dataset_cnf: DatasetInfo = DatasetInfo.model_validate(dataset_info)

    dataloader: DataLoader = get_dataloader(
        # Some repository-owned info files intentionally leave ``repository``
        # empty so that the dataset directory remains relocatable.
        dataset_type=dataset_type,
        dataset_path=dataset_cnf.repository or path,
        dataset_info=dataset_cnf,
        batch=dataset_cnf.batch_size,
        transform=T.Compose([T.ToTensor(), ]),
        num_workers=dataset_cnf.num_workers,
        name=dataset_cnf.name,
        folder_data=dataset_cnf.folder_data,
        parquet_info=dataset_cnf.parquet_info,
    )
    dataset: Dataset = dataloader.dataset
    assert len(dataset) > 0
    sample, label = dataset[0]
    print(label)
    assert isinstance(sample, Tensor)
    assert sample.ndim == 3
    assert isinstance(label, int)
