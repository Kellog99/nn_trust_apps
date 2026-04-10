from pydantic import Field, model_validator

from nn_trust import Task, AttackType, Knowledge
from nn_trust.attack import EvasionAttackFactory
from nn_trust.attack._evasion import EvasionAttack
from nn_trust.attack.normalization import LpNormalization
from ._uap import _UAPAttackConfig, _UAPAttack


class FGSMUAPAttackConfig(_UAPAttackConfig):
    delta: float = Field(
        default=0.1,
        description="Target error rate in classification.",
        ge=0.0,
        le=1.0,
        title="Target error rate"
    )
    lp_norm: LpNormalization = Field(
        default_factory=lambda: LpNormalization(p=2.0, radius=48.0),
        description="Lp normalization for the perturbation.",
    )
    attack: EvasionAttack | None = Field(
        default=None,
        description="Evasion attack algorithm to reach classification boundary."
    )

    @model_validator(mode="after")
    def validate_sub_attack(self):
        if self.attack is None:
            self.attack = EvasionAttackFactory.create(
                class_id="fgsm",
                model=self.model,
                task=self.task,
                device=self.device
            )
        return self


@EvasionAttackFactory.register(
    name="Fast Gradient Sign Method Universal Perturbation Attack",
    description="A fast gradient sign Universal perturbation attack.",
    task={Task.Classification},
    type=AttackType.Digital,
    knowledge=Knowledge.White
)
class FGSMUAPAttack(_UAPAttack):
    CONFIG_T = FGSMUAPAttackConfig
