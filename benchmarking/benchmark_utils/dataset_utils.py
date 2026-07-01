import os
import random
from pathlib import Path

import numpy
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image as PILImage
from torch.utils.data import Subset, DataLoader


class ImageDatasetFolder(torchvision.datasets.ImageFolder):
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

def get_data_transformation_config(
    transform_id: str,
    size: int | None = None,
    crop: int | None = None,
    mean: list[float] | None = None,
    std: list[float] | None = None
):
    """
    An utility function providing the transformation and inverse transformation
    for dataset at hand"""

    if transform_id == "imagenet":
        # mean, std = [0.5074, 0.5308, 0.5306], [0.2639, 0.2518, 0.2521]
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(size=(size, size)),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                # torchvision.transforms.Normalize(mean=[0.5074, 0.5308, 0.5306], std=[0.2639, 0.2518, 0.2521])
            ]
        )
    elif transform_id == "normalize_only":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )
    elif transform_id == "imagenet_like_crop":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(size=(size, size)),
                torchvision.transforms.CenterCrop(crop),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    else:
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(size=(size, size)),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    if transform_id == "imagenet":
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        inverse_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(
                    mean=[-mean_el / std_el for mean_el, std_el in zip(mean, std)], std=[1 / std_el for std_el in std]
                ),
                torchvision.transforms.Resize(size=(size, size)),
                torchvision.transforms.ToPILImage(),
            ]
        )
    if transform_id == "imagenet_like_crop":
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        inverse_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(
                    mean=[-mean_el / std_el for mean_el, std_el in zip(mean, std)], std=[1 / std_el for std_el in std]
                ),
                torchvision.transforms.Resize(size=(size, size)),
                torchvision.transforms.ToPILImage(),
            ]
        )
    else:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        inverse_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(
                    mean=[-mean_el / std_el for mean_el, std_el in zip(mean, std)], std=[1 / std_el for std_el in std]
                ),
                torchvision.transforms.Resize(size=(size, size)),
                torchvision.transforms.ToPILImage(),
            ]
        )

    return transform, inverse_transform



def get_dataloader(
    dataset: str,
    batch: int,
    subset: int,
    type_dataset: int,
    transform: T.Compose,
    num_workers: int = 4,
    name: str | None = None,
    **kwargs,
) -> DataLoader:
    """
    Return the dataloader to use and the inverse transformation to use for displaying the images
    """

    if not os.path.exists(dataset):
        raise ValueError(f"The dataset --------{dataset} does not exists.")
    if type_dataset == 1:
        raise NotImplementedError("This dataset type is not implemented")
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

    if name is not None:
        dataset.name = name
    else:
        dataset.name = Path(dataset).name


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