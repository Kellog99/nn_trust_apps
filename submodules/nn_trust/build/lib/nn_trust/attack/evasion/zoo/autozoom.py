from typing import Any, cast

from pydantic import Field

from ._zoo import _ZooAttack, ZooAttackConfig
from nn_trust import Task, AttackType, Knowledge
from nn_trust.attack import EvasionAttackConfig, EvasionAttackFactory
from nn_trust.attack.gradient_approximations import GradientEstimator, RandomUnitDirectionEstimator


class AutozoomAttackConfig(ZooAttackConfig):
    gradient_estimator: type[GradientEstimator] = Field(
        default=RandomUnitDirectionEstimator,
        description="Type of gradient estimator to be used.",
        # condition=lambda x: x is not None
    )
    gradient_estimator_args: dict[str, Any] = Field(
        default_factory=lambda: dict(n_samples=8),
        description="Parameters for the gradient estimator."
    )


@EvasionAttackFactory.register(
    name="Autoencoder-based Zeroth Order Optimization Method",
    description="Black-box attack leverages efficient gradient estimation through finite differences with additional dimensionality reduction techniques.",
    task={Task.Classification},
    type=AttackType.Digital,
    knowledge=Knowledge.White
)
class AutozoomAttack(_ZooAttack):
    r"""
    This class implement Autozoom attack from https://doi.org/10.48550/arXiv.1805.11770.
    """
    CONFIG_T = AutozoomAttackConfig

    def __init__(self, config: EvasionAttackConfig):
        super().__init__(config)
        self._config = cast(AutozoomAttackConfig, self._config)
