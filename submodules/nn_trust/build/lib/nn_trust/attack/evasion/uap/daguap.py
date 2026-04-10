from pydantic import Field, model_validator

from nn_trust import Task, AttackType, Knowledge
from nn_trust.attack import EvasionAttackFactory
from nn_trust.attack._evasion import EvasionAttack
from nn_trust.attack.normalization import LpNormalization
from ._uap import _UAPAttackConfig, _UAPAttack


class DAGUAPAttackConfig(_UAPAttackConfig):
    delta: float = Field(
        default=0.1,
        description="Target error rate in classification.",
        ge=0.0,
        le=1.0
    )
    lp_norm: LpNormalization = Field(
        default_factory=lambda: LpNormalization(p=2.0, radius=48.0),
        description="Lp normalization for the perturbation.",
    )
    attack: EvasionAttack | None = Field(None, description="Evasion attack algorithm to reach classification boundary.")

    @model_validator(mode="after")
    def validate_sub_attack(self):
        if self.attack is None:
            self.attack = EvasionAttackFactory.create(
                class_id="dag",
                model=self.model,
                task=self.task,
                device=self.device
            )
        return self


@EvasionAttackFactory.register(
    name="Dense Adversary Generation Universal Perturbation Attack",
    description="A dag universal Attack.",
    task={Task.Classification},
    type=AttackType.Digital,
    knowledge=Knowledge.White
)
class DAGUAPAttack(_UAPAttack):
    CONFIG_T = DAGUAPAttackConfig


class FUAPAttackConfig(_UAPAttackConfig):
    delta: float = Field(
        default=0.01,
        description="Target error rate in classification.",
        ge=0.0,
        le=1.0
    )
    lp_norm: LpNormalization = Field(
        default_factory=lambda: LpNormalization(p=2.0, radius=48.0),
        description="Lp normalization for the perturbation.",
    )
    attack: EvasionAttack | None = Field(None, description="Evasion attack algorithm to reach classification boundary.")

    epsilon: float = Field(
        default=0.300449,
        description="Force of the attack.",
        gt=0,
    )

    @model_validator(mode="after")
    def validate_sub_attack(self):
        if self.attack is None:
            self.attack = EvasionAttackFactory.create(
                class_id="deepfool",
                model=self.model,
                task=self.task,
                device=self.device
            )
        return self


@EvasionAttackFactory.register(
    name="Fast Universal Adversarial Perturbation Attack (FUAP)",
    description="A white-box universal adversarial attack utilizing a faster loss computation than UAP with minimal drawbacks in efficancy.",
    task={Task.Classification},
    type=AttackType.Digital,
    knowledge=Knowledge.White
)
class FUAPAttack(_UAPAttack):
    CONFIG_T = FUAPAttackConfig
