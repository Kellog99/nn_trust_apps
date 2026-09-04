from io import BytesIO
from pathlib import Path

import requests
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from nn_trust import Knowledge, Task
from nn_trust.attack._cv import CVModelAdapter


def _fetch_dog_images(num_images: int) -> list[Image.Image]:
    """Fetch images from the Dog CEO API. Raises on network/HTTP failure."""
    resp = requests.get(
        url=f"https://dog.ceo/api/breeds/image/random/{num_images}",
        timeout=10,
    )
    resp.raise_for_status()
    urls = resp.json()["message"]  # always a list from the .../random/{n} endpoint

    images = []
    for url in urls:
        img_resp = requests.get(url, timeout=10)
        img_resp.raise_for_status()
        images.append(Image.open(BytesIO(img_resp.content)))
    return images


def get_dog_image(
        num_images: int = 1,
        cache_dir: Path = Path("~/Desktop/lavoro/cached_images").expanduser().resolve(),
) -> Image.Image | list[Image.Image]:
    """
    Return dog images, preferring a local cache; fetch and cache any shortfall.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    if num_images < 1:
        raise ValueError("num_images must be non-negative")

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_files = sorted(cache_dir.glob("*.jpg"))

    images = [Image.open(f) for f in cached_files[:num_images]]

    missing = num_images - len(images)
    if missing > 0:
        fetched = _fetch_dog_images(missing)
        start_idx = len(cached_files)
        for i, img in enumerate(fetched):
            path = cache_dir / f"dog_{start_idx + i:04d}.jpg"
            img.save(path, format="JPEG")
        images.extend(fetched)

    return images[0] if num_images == 1 else images


def get_dummy_cv_model() -> CVModelAdapter:
    model: torch.nn.Module = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    return CVModelAdapter(
        model=model,
        threat_model=Knowledge.White,
        task=Task.Classification,
    )


class PILImageDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        return image, 0  # dummy label, evaluation expects (x, y) pairs


def get_dummy_dataloader(num_samples: int = 3) -> DataLoader:
    """
    Return a mock dataloader composed of dog images
    :param num_samples:
    :return:
    """

    # Usage
    images: list[Image.Image] = get_dog_image(num_images=num_samples)
    dataset = PILImageDataset(images, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]))

    return DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True
    )
