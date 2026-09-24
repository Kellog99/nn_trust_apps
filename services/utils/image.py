import base64
import io

import torch
from PIL import Image
from torchvision.transforms import v2 as T


def b64str_to_pil(b64_image_str: str) -> Image.Image:
    image_bytes = base64.b64decode(b64_image_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def tensor_image_to_b64str(image: torch.Tensor) -> str:
    """Convert a CHW or single-image BCHW tensor to a Base64 PNG string."""
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(
                f"Expected a batch of size 1, got shape {tuple(image.shape)}"
            )
        image = image[0]

    if image.ndim != 3:
        raise ValueError(f"Expected shape (C, H, W), got {tuple(image.shape)}")

    image = torch.nan_to_num(
        image.detach().cpu(), nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
    pil_img = T.ToPILImage()(image)

    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
