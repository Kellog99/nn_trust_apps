from typing import Literal, Optional

from pydantic import BaseModel, Field, ConfigDict

from models.info import ModelInfo
from models.model import RegisteredObject


class SingleAttackProps(BaseModel):
    input: str
    attack: RegisteredObject
    model: ModelInfo


class JailbreakAttackProps(BaseModel):
    input: str
    attack: RegisteredObject
    model: ModelInfo
    attacker: Optional[ModelInfo] = None
    judge: Optional[ModelInfo] = None
    max_new_tokens: Optional[int] = 2048

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


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

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class Bubble(BaseModel):
    sender: Literal["user", "model"]
    msg: str
    score: Optional[float] = None


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
