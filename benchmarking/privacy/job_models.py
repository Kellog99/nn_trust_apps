"""App-side privacy job models."""

from enum import Enum
from pathlib import Path
from typing import Any, Literal, Union
from pydantic_core import core_schema

from pydantic import BaseModel, ConfigDict, Field, model_validator
from nn_trust.attack.utils.training import OptimizerType


class RuntimeOptionConfig(BaseModel):
    """Runtime options for one privacy execution."""

    overwrite: bool = False
    output_path: str
    output_format: str = "json"
    gpu: bool = True
    mode: Literal["local_serial"] = "local_serial"
    load_results: bool = False
    num_images_to_save: int = 5
    save_perturbation: bool = False


class PrivacyProtocolId(str, Enum):
    MEMBERSHIP_INFERENCE = "membership_inference"
    PROPERTY_INFERENCE = "property_inference"
    MODEL_INVERSION = "model_inversion"
    RECONSTRUCTION = "reconstruction"


class MaterializedPrivacySplits(BaseModel):
    shadow: list[int] = Field(default_factory=list)
    target_train: list[int] = Field(default_factory=list)
    target_val: list[int] = Field(default_factory=list)
    target_test: list[int] = Field(default_factory=list)

    @property
    def dataset_size(self) -> int:
        return sum(len(getattr(self, f)) for f in ("shadow", "target_train", "target_val", "target_test"))

    @property
    def target_pool_size(self) -> int:
        return len(self.target_train) + len(self.target_val) + len(self.target_test)


class PrivacySplitStrategy(str, Enum):
    LEGACY_COMPAT = "legacy_compat"
    EXPLICIT_RATIOS = "explicit_ratios"


class TargetModelSourceType(str, Enum):
    CHECKPOINT = "checkpoint"
    TRAIN = "train"


class PrivacyDatasetConfig(BaseModel):
    """Dataset source description for privacy jobs."""

    dataset_id: str
    root: Path
    task_attr: str | None = None
    use_embeddings: bool | None = None
    max_samples: int | None = Field(default=None, ge=4)
    seed: int = Field(default=42)


class PrivacySplitPlanConfig(BaseModel):
    """Dataset partitioning policy shared across privacy protocols.

    ``legacy_compat`` reproduces the current legacy privacy dataset wrapper's
    partition semantics so checkpoint-backed migrations can preserve target
    membership provenance. ``explicit_ratios`` is reserved for app-native flows
    where target validation/test ratios are intentionally caller-defined.
    """

    strategy: PrivacySplitStrategy = Field(default=PrivacySplitStrategy.LEGACY_COMPAT)
    shadow_ratio: float = Field(default=0.5, ge=0.1, le=0.9)
    target_train_ratio: float = Field(default=0.8, gt=0.0, lt=1.0)
    target_val_ratio: float | None = Field(default=None, ge=0.0, lt=1.0)
    sort_indices_within_split: bool = Field(
        default=False,
        description=(
            "Whether to return split indices in ascending dataset-index order. Disable to preserve "
            "the randomized partition order induced by the split seed."
        ),
    )
    seed: int = Field(default=42)

    @model_validator(mode="after")
    def validate_target_pool_ratios(self) -> "PrivacySplitPlanConfig":
        if self.strategy == PrivacySplitStrategy.LEGACY_COMPAT:
            if self.target_val_ratio is not None:
                raise ValueError(
                    "target_val_ratio must be omitted when strategy='legacy_compat', "
                    "because val/test are derived from the legacy remainder split."
                )
            return self

        if self.target_val_ratio is None:
            raise ValueError("target_val_ratio is required when strategy='explicit_ratios'.")
        if self.target_train_ratio + self.target_val_ratio >= 1.0:
            raise ValueError(
                "target_train_ratio + target_val_ratio must be < 1.0 so a target test split remains."
            )
        return self


class PrivacyTargetTrainingOverrideConfig(BaseModel):
    """Optional per-job trainer overrides layered on top of one recipe."""

    epochs: int | None = Field(default=None, ge=1)
    batch_size: int | None = Field(default=None, ge=1)
    learning_rate: float | None = Field(default=None, gt=0.0)
    weight_decay: float | None = Field(default=None, ge=0.0)
    momentum: float | None = Field(default=None, ge=0.0, le=1.0)
    optimizer: OptimizerType | None = None
    seed: int | None = None

    def to_update_mapping(self) -> dict[str, Any]:
        """Return only the explicitly configured trainer fields.

        Note: the returned mapping may include ``optimizer`` when explicitly set
        in the override config.  Callers that layer overrides on top of a recipe
        should be aware that the recipe's optimizer choice will be silently
        replaced.  This is intentional for property-inference shadow models
        (which always use Adam) but may be surprising for other recipes if the
        override config inadvertently sets ``optimizer``.
        """
        payload = self.model_dump(exclude_none=True)
        if "scheduler_params" in payload and not payload["scheduler_params"]:
            payload.pop("scheduler_params")
        return payload


class CheckpointTargetModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    source_type: Literal[TargetModelSourceType.CHECKPOINT] = TargetModelSourceType.CHECKPOINT
    model_id: str
    checkpoint_path: Path
    shadow_model_id: str | None = None
    training_recipe_id: Literal[None] = None
    property_ratio: Literal[None] = None
    property_name: Literal[None] = None
    property_target_ratio: Literal[None] = None
    training_overrides: Literal[None] = None


class TrainTargetModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    source_type: Literal[TargetModelSourceType.TRAIN] = TargetModelSourceType.TRAIN
    model_id: str
    training_recipe_id: str
    checkpoint_path: Literal[None] = None
    property_ratio: Literal["low", "high"] | None = None
    property_name: str | None = None
    property_target_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    shadow_model_id: str | None = None
    training_overrides: PrivacyTargetTrainingOverrideConfig | None = None

    @model_validator(mode="after")
    def validate_property_fields(self) -> "TrainTargetModelConfig":
        if self.property_ratio is not None:
            if self.property_name is None:
                raise ValueError("property_name is required when property_ratio is set.")
            if self.property_target_ratio is None:
                raise ValueError("property_target_ratio is required when property_ratio is set.")
        return self


class PrivacyTargetModelConfig:
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> core_schema.CoreSchema:
        return core_schema.union_schema([
            CheckpointTargetModelConfig.__pydantic_core_schema__,
            TrainTargetModelConfig.__pydantic_core_schema__,
        ])

    def __new__(cls, **kwargs: Any) -> CheckpointTargetModelConfig | TrainTargetModelConfig:
        source_type = kwargs.get("source_type", TargetModelSourceType.CHECKPOINT)
        data = {k: v for k, v in kwargs.items() if v is not None}
        if source_type == TargetModelSourceType.CHECKPOINT or source_type == "checkpoint":
            return CheckpointTargetModelConfig(**data)
        return TrainTargetModelConfig(**data)


class PrivacyAttackPayload(BaseModel):
    """Attack identifier plus attack-local parameters."""

    attack_id: str
    attack_params: dict[str, Any] = Field(default_factory=dict)


class PrivacyJobConfig(BaseModel):
    dataset: PrivacyDatasetConfig
    split_plan: PrivacySplitPlanConfig = Field(default_factory=PrivacySplitPlanConfig)
    target_model: PrivacyTargetModelConfig
    attack: PrivacyAttackPayload
    options: RuntimeOptionConfig
    verbose: bool = False
