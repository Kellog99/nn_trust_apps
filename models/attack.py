from typing import Literal, Optional, Self

import torch
from pydantic import BaseModel, Field, ConfigDict, model_validator

from models.info import ModelInfo
from models.model import RegisteredObject
from services.utils.image import tensor_image_to_b64str


class SingleAttackProps(BaseModel):
    input: str
    device: Literal["cpu", "gpu", "cuda", "mps"] = "gpu"
    attack: RegisteredObject
    model: ModelInfo

    def resolve_device(self) -> torch.device:
        """Resolve an API device name to an available PyTorch device.

        ``gpu`` is backend-agnostic: CUDA is preferred when available, then
        Apple MPS. Requests for an unavailable accelerator safely fall back to
        the CPU.
        """
        if self.device in {"gpu", "cuda"} and torch.cuda.is_available():
            return torch.device("cuda")
        if self.device in {"gpu", "mps"} and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


class SingleAttackOutput(BaseModel):
    """
    This model has the goal to send the information to the frontend regarding one image attack
    """
    x_adv: str | torch.Tensor
    adv_perturbation: str | torch.Tensor
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

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="after")
    def tensors_to_base64(self) -> Self:
        if isinstance(self.x_adv, torch.Tensor):
            self.x_adv = tensor_image_to_b64str(self.x_adv)
        if isinstance(self.adv_perturbation, torch.Tensor):
            self.adv_perturbation = tensor_image_to_b64str(self.adv_perturbation)
        return self
