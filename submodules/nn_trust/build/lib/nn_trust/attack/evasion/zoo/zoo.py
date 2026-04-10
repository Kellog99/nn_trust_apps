from typing import cast

from nn_trust import Knowledge, AttackType, Task
from nn_trust.attack import EvasionAttackFactory
from ._zoo import _ZooAttack, ZooAttackConfig


@EvasionAttackFactory.register(
    name="Zeroth-Order Optimization Attack",
    description="Black-box attack leverages efficient gradient estimation through finite differences for adversarial exploitation.",
    task={Task.Classification},
    type=AttackType.Digital,
    knowledge=Knowledge.White
)
class ZooAttack(_ZooAttack):
    r"""
    This class implement Zoo attack from https://doi.org/10.1145/3128572.3140448.
    """
    CONFIG_T = ZooAttackConfig

    def __init__(self, config: ZooAttackConfig):
        super().__init__(config)
        self._config = cast(ZooAttackConfig, self._config)
