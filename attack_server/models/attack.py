from typing import Self

import torch
from pydantic import BaseModel, model_validator, Field
from torchvision.transforms import v2 as T

from attack_server.lib.model import RegisteredObject
from attack_server.models.info import ModelInfo
from attack_server.utils.utils import pil_to_b64str


class SingleAttackProps(BaseModel):
    image: str
    attack: RegisteredObject
    model: ModelInfo


class SingleAttackOutput(BaseModel):
    """
    This model has the goal to send the information to the frontend regarding one image attack
    """
    x_adv: torch.Tensor | str
    adv_perturbation: torch.Tensor | str
    original_prediction: str
    adversarial_prediction: str
    advance_metrics: dict[str, float]
    confidence: dict[str, dict] = Field(
        default={
            "adversarial": {},
            "original": {}
        },
        description="This contain original and adversarial predictions' confidence."
    )

    class Config:
        arbitrary_types_allowed = True  # Required for torch.Tensor

    @model_validator(mode="after")
    def image_to_base64(self) -> Self:
        """
        Since the elements have to go to the frontend, then the image must be base64 encoded.
        """
        for atr in ["x_adv", "adv_perturbation"]:
            img = getattr(self, atr, None)
            if isinstance(img, torch.Tensor):
                pil_img = T.ToPILImage()(img.squeeze())
                setattr(self, atr, pil_to_b64str(pil_img))
        return self
