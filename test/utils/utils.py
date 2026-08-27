from io import BytesIO

import requests
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from nn_trust import Knowledge, Task
from nn_trust.attack._cv import CVModelAdapter


def get_dog_image() -> Image.Image:
    """Fetch a random dog image from the Dog CEO API."""
    resp = requests.get("https://dog.ceo/api/breeds/image/random", timeout=10)
    resp.raise_for_status()
    url = resp.json()["message"]

    img_resp = requests.get(url, timeout=10)
    img_resp.raise_for_status()
    return Image.open(BytesIO(img_resp.content))


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
    images: list[Image.Image] = [get_dog_image() for _ in range(num_samples)]
    dataset = PILImageDataset(images, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]))

    return DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True
    )
