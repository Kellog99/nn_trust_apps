from typing import Literal, Optional, Self

import torch
from pydantic import BaseModel, Field, ConfigDict, model_validator

from models.info import ModelInfo
from models.model import RegisteredObject
from services.utils.image import tensor_image_to_b64str


class SingleAttackProps(BaseModel):
    input: Optional[str] = None
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


class JailbreakAttackProps(SingleAttackProps):
    goal: str
    attacker: Optional[ModelInfo] = None
    judge: Optional[ModelInfo] = None
    max_new_tokens: Optional[int] = 4096
    n_ctx: Optional[int] = 8192

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="after")
    def validate_judge(self) -> Self:
        if self.attacker is None:
            self.attacker = self.model
        if self.judge is None:
            self.judge = self.model
        return self


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


class Bubble(BaseModel):
    sender: Literal["user", "model"]
    msg: str
    score: Optional[float] = None


class JailbreakHistoryEntry(BaseModel):
    """Lightweight metadata for a saved jailbreak attack run, used to populate the history board."""
    id: str
    goal: str
    success: bool
    best_score: Optional[float] = None
    n_attempts: int
    saved_at: str


class JailbreakAttackOutput(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), arbitrary_types_allowed=True)
    goal: str
    success: bool
    best_prompt: str
    best_response: str
    best_score: float
    history: list[dict]
    conversations: Optional[list[list[dict]]] = None
    metadata: dict
    adversarial_prompt: Optional[str] = None
    model_response: Optional[str] = None
    advance_metrics: Optional[dict[str, float]] = None
