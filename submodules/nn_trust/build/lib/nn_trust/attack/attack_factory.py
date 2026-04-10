from typing import Type

from pydantic import Field, field_validator

from nn_trust import Knowledge
from nn_trust.attack import EvasionAttack
from nn_trust.core import AttackType, Task
from nn_trust.factory import Info, Factory


class AttackInfo(Info[EvasionAttack]):
    type: AttackType = Field(
        default=...,
        description="This field represent the attack's typology , i.e. physical or digital."
    )
    knowledge: Knowledge = Field(
        default=...,
        description="This field represent the required knowledge, i.e. White or a Black box."
    )

    @field_validator('class_type')
    @classmethod
    def validate_attack_class(cls, v: Type[EvasionAttack]) -> Type[EvasionAttack]:
        """Validate attack class inheritance."""
        # Check if it's a class (type)
        if not isinstance(v, type):
            raise ValueError("Attack must be a class, not an instance")
        if not hasattr(v, 'CONFIG_T'):
            raise ValueError("Attack must have the class variable `CONFIG_T` for the configuration file.")
        return v


class EvasionAttackFactory(Factory):
    """
    Factory for creating evasion attacks.
    """
    _info_type = AttackInfo

    @classmethod
    def filter(cls,
               info: AttackInfo,
               task: set[Task],
               attack_type: list[AttackType] | None = None,
               knowledge: list[Knowledge] | None = None,
               **kwargs) -> bool:
        """
            This method implements the filtering logic for the attacks.
        """
        in_task: bool = super().filter(info=info, task=task)
        in_type: bool = info.type in attack_type if attack_type else True

        in_knowledge: bool = info.knowledge in knowledge if knowledge else True

        return in_task and in_type and in_knowledge
