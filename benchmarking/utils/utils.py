import os
import random
from pathlib import Path

import numpy
import timm
from PIL import Image as PILImage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50

from nn_trust.core import ModelAdapter
from .imagenet2012_loader import ImageNetTrainDataset
from .model_library import models_library


class ImageDatasetFolder(ImageFolder):
    def __init__(self, root: str, transform=None, target_transform=None, is_valid_file=None):
        super().__init__(root=root, transform=transform, target_transform=target_transform, is_valid_file=is_valid_file)
        self.data_root = root

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = PILImage.open(path).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        element_info = {"path": str(Path(path).relative_to(self.data_root))}

        return sample, target, element_info


def get_structure(path: Path | str) -> dict:
    path = path if isinstance(path, Path) else Path(path)
    if path.is_dir():
        out = {element: get_structure(path=path / element) for element in os.listdir(path) if (path / element).is_dir()}
        out["files"] = [element for element in os.listdir(path) if not (path / element).is_dir()]

        return out
    else:
        return {"files": path.name}


def get_dataloader(
    dataset: str,
    batch: int,
    subset: int,
    type_dataset: int,
    transform: transforms.Compose,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """
    Return the dataloader to use and the inverse transformation to use for displaying the images
    """
    if not os.path.exists(dataset):
        raise ValueError("The dataset does not exists.")
    if type_dataset == 1:
        # get the transformation associated with the model
        # and its inverse for the display
        dataset = ImageNetTrainDataset(data_root=dataset, data_transform=transform)

    elif type_dataset == 2:

        def check_valid_image(filename: str):
            try:
                with PILImage.open(filename) as img:
                    img.verify()  # Verify that the image is valid
                return True
            except Exception:
                return False

        dataset = ImageDatasetFolder(dataset, transform=transform, is_valid_file=check_valid_image)
    else:
        raise ValueError(f"The type of the dataset {type_dataset} is not valid.")

    if subset is None or subset < 0:
        subset = list(range(len(dataset)))
    else:
        subset = list(range(min(subset, len(dataset))))
    subdataset = Subset(dataset, subset)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(1234)

    dataloader = DataLoader(
        subdataset,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
    )

    return dataloader


def get_model(
    model: ModelAdapter | torch.nn.Module = None,
    model_name: str = None,
    model_type: str = None,
    model_task: str = None,
    model_weights_path: str | Path = None,
    mean: float | list[float] = 0.5,
    std: float | list[float] = 0.5,
) -> ModelAdapter:
    """
    In this function it is set the model in the correct form,
    independent from the starting point
    """
    tt = transforms.Compose(
        [
            transforms.Normalize(mean=[-1, -1, -1], std=[2.0, 2.0, 2.0]),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    if model:
        model = ModelAdapter(model, name=model._get_name(), task=model_task)
    elif model_name and model_type == "timm":
        model = ModelAdapter(model=timm.create_model(model_name, pretrained=True), name=model_name, transform=tt, task=model_task)
    elif model_name and model_type == "saved_model":
        model = ModelAdapter(model=torch.load(model_weights_path, weights_only=False), name=model_name, transform=tt, task=model_task)
    elif model_name and model_type == "saved_weights":
        # model = ResNet50Dirichlet()
        model = models_library[model_name]()
        state_dict = torch.load(model_weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model = ModelAdapter(model=model, name=model_name, transform=tt, task=model_task)
    else:
        raise ValueError("You must provide a model or a model name and type.")

    model.model.eval()
    return model


def config_file_path_selector(config_dir: Path | str = ".") -> Path:
    """Seletc a YAML configuration file from the script directory."""
    config_files = [f for f in os.listdir(config_dir) if f.endswith(".yaml") or f.endswith(".yml")]
    if not config_files:
        raise FileNotFoundError("No YAML configuration files found in the script directory.")
    print("Available configuration files:")
    for idx, fname in enumerate(config_files):
        print(f"{idx}: {fname}")
    selected_idx = int(input("Select configuration file by index: "))
    if selected_idx < 0 or selected_idx >= len(config_files):
        raise IndexError("Selected index is out of range.")
    selected_config_path = config_dir / config_files[selected_idx]
    return selected_config_path
