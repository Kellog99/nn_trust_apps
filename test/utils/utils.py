from io import BytesIO

import requests
import torch
from PIL import Image
from torchvision import models

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
