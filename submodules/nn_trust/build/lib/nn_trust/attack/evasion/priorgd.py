from typing import Optional

import torch.nn as nn
from pydantic import Field

from nn_trust import Task, AttackType, Knowledge
from nn_trust.attack import EvasionAttackFactory
from nn_trust.attack.evasion._numerical import _NumericalMethodsAttack, _NumericalMethodsAttackConfig
from nn_trust.attack.normalization import Normalization
from nn_trust.loss.loss_composer import LossComposer


class PriorGDAttackConfig(_NumericalMethodsAttackConfig):
    loss: nn.Module = Field(
        default_factory=lambda: LossComposer(losses={"misclassification": {}}),
        description="Loss to minimise in the PGD attack.",
    )
    gradient_normalizer: Optional[Normalization] = Field(
        default=None, description="Gradient normalizer to use during the attack."
    )


@EvasionAttackFactory.register(
    name="Prior and Gradient Descent",
    description="A white box attack that uses additional prior knowledge of the image to be more efficient.",
    task={Task.Classification},
    type=AttackType.Digital,
    knowledge=Knowledge.White
)
class PriorGDAttack(_NumericalMethodsAttack):
    r"""Simlar to Projected Gradient Descent, except the prior replaces the role
    of the projection in the optimization procedure allowing for pertrurbation in the attack
    that are less visible than a mere clamping in values.

    The prior loss is defined as
    .. math::
        \text{TotalVariation}(\hat{x}) + \| \hat{x}-x \|_2 + (\hat{x} - 1)^+ + (- \hat{x})^+ + \sum_{i=1}^N 1_{\{\hat{x}_i \neq x_i\}}

    with :math:`\hat{x}` being the adversarial image and :math:`x` being the original image.


    :param config: The configuration can be tweaked via changing the following parameters:
        1. ``optimizer`` which specifies a type of :class:`torch.optim.Optimizer`,
        2. ``optimizer_params`` a dictionary that specifies the parameters of the given ``optimizer``.
        3. ``scheduler`` which specifies a type of :class:`torch.optim.lr_scheduler.LRScheduler` as further optimization
            procedure to tweak iteratively the learning rate of the given ``optimizer``.
        4. ``scheduler_params`` a dictionary that specifies the parameters of the given ``scheduler``.
        5. ``max_iters`` the number of iteration procedure to compute.

    .. Note: The default configuration uses :class:`torch.optim.Adam` with learning rate `1e-3`.

    Example::

    Consider a batch of images :math:`(B, C, H, W)`, which we denote as ``data_input``, and a corresponding
    one-hot encoded labels ``target_label`` of shape :math:`(B, N_c)`. Let ``model`` be a :class:`AttackedModel`.
    Then, we can use the attack as follows

    >>> from nn_trust.attack import EvasionAttackFactory
    >>> cnf = EvasionAttackFactory.get_config("priorgd", model=model, task=Task.Classification, targeted=True, max_iters=20)
    >>> atk = EvasionAttackFactory.create_attack(config=cnf)
    >>> atk.generate(data_input, target_label)

    """

    CONFIG_T = PriorGDAttackConfig
