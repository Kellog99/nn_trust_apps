from typing import Any, Literal, Self

import torch
from pydantic import BaseModel, Field, model_validator
from torchvision.transforms import v2 as T

from models.model import RegisteredObject
from services.utils.utils import pil_to_b64str


class PrivacyDatasetRef(BaseModel):
    id: str
    root: str | None = None
    task_attr: str | None = None
    use_embeddings: bool | None = None
    max_samples: int | None = None
    seed: int = 42


class PrivacyModelRef(BaseModel):
    id: str
    source_type: Literal["train", "checkpoint"] = "train"
    training_recipe_id: str | None = "classification_default"
    checkpoint_path: str | None = None
    shadow_model_id: str | None = None
    property_ratio: Literal["low", "high"] | None = None
    property_name: str | None = None
    property_target_ratio: float | None = None
    training_overrides: dict | None = None


class PrivacyAttackProps(BaseModel):
    attack: RegisteredObject
    dataset: PrivacyDatasetRef
    model: PrivacyModelRef


class PrivacyArtifactRef(BaseModel):
    artifact_id: str
    filename: str
    media_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PrivacyAttackOutput(BaseModel):
    metrics: dict[str, Any]
    reconstructions: list[torch.Tensor | str] | None = None
    artifacts: list[PrivacyArtifactRef] = Field(default_factory=list)
    attack_metadata: dict[str, Any] = Field(default_factory=dict)
    target_metadata: dict[str, Any] = Field(default_factory=dict)
    dataset_metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def image_to_base64(self) -> Self:
        if self.reconstructions is None:
            return self
        out: list[str] = []
        for image in self.reconstructions:
            if isinstance(image, torch.Tensor):
                out.append(pil_to_b64str(T.ToPILImage()(image.squeeze())))
            else:
                out.append(image)
        self.reconstructions = out
        return self
