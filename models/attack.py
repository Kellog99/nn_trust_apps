from typing import Literal, Optional

from pydantic import BaseModel, Field

from models.info import ModelInfo
from models.model import RegisteredObject


class SingleAttackProps(BaseModel):
    input: str
    attack: RegisteredObject
    model: ModelInfo


class SingleAttackOutput(BaseModel):
    """
    This model has the goal to send the information to the frontend regarding one image attack
    """
    x_adv: str
    adv_perturbation: str
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


class Bubble(BaseModel):
    sender: Literal["user", "model"]
    msg: str
    score: Optional[float] = None


class JailbreakAttackOutput(BaseModel):
    adversarial_prompt: str
    conversations: Optional[list[list[Bubble]]] = None
    model_response: str
    advance_metrics: dict[str, float]
