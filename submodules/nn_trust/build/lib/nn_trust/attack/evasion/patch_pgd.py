from typing import Optional

import torch.nn as nn
from pydantic import Field, field_validator

from nn_trust import Task, AttackType, Knowledge
from nn_trust.attack import EvasionAttackFactory
from nn_trust.attack.evasion._numerical import _NumericalMethodsAttack, _NumericalMethodsAttackConfig
from nn_trust.attack.normalization import LpNormalization, Normalization
from nn_trust.loss.loss_composer import LossComposer


class PatchPGDAttackConfig(_NumericalMethodsAttackConfig):
    loss: nn.Module = Field(
        default_factory=lambda: LossComposer(losses={"misclassification": {}}),
        description="Loss function to use for the attack.",
    )
    gradient_normalizer: Normalization = Field(
        default_factory=lambda: LpNormalization(p=2.0, radius=16.0),
        description="Gradient normalizer to use during the attack.",
    )
    patch: Optional[tuple[int, int]] = Field(
        default=(64, 64),
        description="If the adversarial is a patch or a perturbation."
    )
    image_smoothing: bool = Field(
        default=False,
        description="If the perturbation should be created as a patch of the original domain of the images.",
    )

    @field_validator("loss")
    def valid_loss(cls, v):
        if not isinstance(v, nn.Module):
            raise ValueError("loss must be an instance of nn.Module.")
        return v

    @field_validator("gradient_normalizer")
    def valid_gradient_normalizer(cls, v):
        if not isinstance(v, Normalization):
            raise ValueError("gradient_normalizer must be an instance of Normalization.")
        return v


@EvasionAttackFactory.register(
    name="Patch Projected Gradient Descent (PPGD)",
    description="A white-box attack that uses PGD to find a patch-like modification to the original image.",
    task={Task.Classification},
    type=AttackType.Digital,
    knowledge=Knowledge.White
)
class PatchPGDAttack(_NumericalMethodsAttack):
    r"""Implements a class of white-box adversarial attacks whose common factor
    is that we use a projection on the :math:`L^p` sphere instead of the full
    gradient to find the optimal perturbation (see [1]_ for an in-depth analysis).

    :param config: The configuration can be tweaked via changing the following parameters:
        1. ``optimizer`` which specifies a type of :class:`torch.optim.Optimizer`,
        2. ``optimizer_params`` a dictionary that specifies the parameters of the given ``optimizer``.
        3. ``scheduler`` which specifies a type of :class:`torch.optim.lr_scheduler.LRScheduler` as further optimization
            procedure to tweak iteratively the learning rate of the given ``optimizer``.
        4. ``scheduler_params`` a dictionary that specifies the parameters of the given ``scheduler``.
        5. ``max_iters`` the number of iteration procedure to compute.
        6. ``gradient_normalizer.p`` specifies the norm in the :math:`L^p` space.
        7. ``gradient_normalizer.radius`` specifies the maximum allowed norm in the given ``L^p`` space.

        Note: The default configuration uses :class:`torch.optim.Adam` with
        learning rate `1e-3` and ``gradient_normalizer.p = 2``, ``gradient_normalizer.radius = 4.0``.

    Example::

    Consider a batch of images :math:`(B, C, H, W)`, which we denote as ``data_input``, and a corresponding
    one-hot encoded labels ``target_label`` of shape :math:`(B, N_c)`. Let ``model`` be a :class:`AttackedModel`.
    Then, we can compute the Projected Gradient Descent (PGD) attack with respect to the :math:`L^\infty`-norm
    with radius :math:`0.03` as:

    >>> from nn_trust.attack import EvasionAttackFactory
    >>> cnf = EvasionAttackFactory.get_config("patchpgd", model=model, task=Task.Classification, targeted=True, max_iters=20)
    >>> cnf.gradient_normalizer.p = float("inf")
    >>> cnf.gradient_normalizer.radius = 0.03
    >>> atk = EvasionAttackFactory.create_attack(config=cnf)
    >>> atk.generate(data_input, target_label)

    .. [1] Madry et al. “Towards Deep Learning Models Resistant to Adversarial Attacks.” ICLR 2018. ArXiv
    """

    CONFIG_T = PatchPGDAttackConfig
