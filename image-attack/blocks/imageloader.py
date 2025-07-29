from typing import Literal, Optional

import gradio as gr
import torch
import torchvision.transforms.functional as F

from nn_trust import ModelAdapter

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

STANDARD_MEAN = torch.tensor([0.5, 0.5, 0.5])
STANDARD_STD = torch.tensor([0.5, 0.5, 0.5])


def from_image_to_standardized_image(x: torch.Tensor):
    """Rescales a tensor image from the range [0, 1] to [-1, 1]."""
    x = F.normalize(x, STANDARD_MEAN, STANDARD_STD, inplace=False)
    return x


def from_standardized_image_to_resnet(x: torch.Tensor) -> torch.Tensor:
    """Rescales an image from the range [-1, 1] to the appropriate range for
    a ResNet model pre-trained on imagenet with IMAGENET_MEAN and IMAGENET_STD.
    """
    x = F.normalize(x, -STANDARD_MEAN / STANDARD_STD, 1 / STANDARD_STD, inplace=False)
    x = F.normalize(x, IMAGENET_MEAN, IMAGENET_STD, inplace=False)
    return x


def from_standardized_image_to_image(x: torch.Tensor) -> torch.Tensor:
    """Rescales an image from the range [-1, 1] to the range [0, 1]."""
    x = F.normalize(x, -STANDARD_MEAN / STANDARD_STD, 1 / STANDARD_STD, inplace=False)
    return x


def rescale_image_range(x: torch.Tensor) -> torch.Tensor:
    """Rescale tensor values in [0, 1] range."""
    x_min, x_max = x.amin(), x.amax()
    return (x - x_min) / (x_max - x_min)


class InputImage:
    def __init__(self, model: ModelAdapter, device: torch.device, labels_id: Optional[dict[int, str]] = None):
        self.model = model
        self.device = device
        self.img = None
        self.input_model = None
        self.prediction = None
        self.labels_id = labels_id

    def get_image(self, format: Literal["standardized", "pil"] = "standardized"):
        if format == "standardized":
            return self.img
        elif format == "pil":
            return F.to_pil_image(from_standardized_image_to_image(self.img).squeeze(0))
        else:
            raise ValueError(f"The current format '{format}' is not supported")

    def get_prediction(self, format: Literal["id", "label"] = "id"):
        if format == "id":
            return self.prediction
        elif format == "label":
            return self.labels_id.get(self.prediction.item(), "Unknown class").upper()
        else:
            raise ValueError(f"The current format '{format}' is not supported")

    def set_image(self, img):
        # Correctly load a numpy image to a torch tensor
        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1)
        img = F.resize(img, size=(256, 256))
        img = F.center_crop(img, (224, 224))
        img = from_image_to_standardized_image(img)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        # Finally update prediction and image
        self.img = img
        self.prediction = self.model(self.img.to(self.device)).argmax(-1)

    def generate(self):
        gr.Markdown("### 1. Image 📷")
        input_image = gr.Image(
            type="numpy", label="Input Image", sources=["upload", "clipboard"], width=448, height=448, format="png"
        )

        input_image.upload(fn=self.set_image, inputs=input_image)

        return input_image
