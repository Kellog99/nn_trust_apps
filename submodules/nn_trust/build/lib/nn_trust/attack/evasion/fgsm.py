import torch.nn as nn
from pydantic import Field, field_validator
from typing import Literal

from nn_trust import Task, AttackType, Knowledge
from nn_trust.attack import EvasionAttackFactory
from nn_trust.attack.evasion._numerical import _NumericalMethodsAttack, _NumericalMethodsAttackConfig
from nn_trust.attack.normalization import Normalization, SignNormalization
from nn_trust.loss.loss_composer import LossComposer


class FGSMAttackConfig(_NumericalMethodsAttackConfig):

    max_iters: int = Field(
        default=1,
        description="Maximum number of optimization step.",
        ge=1
    )

    loss: nn.Module = Field(
        default_factory=lambda: LossComposer(losses={"misclassification": {}}),
        description="The loss function to use for the attack.",
    )
    gradient_normalizer: Normalization = Field(
        default_factory=lambda: SignNormalization(),
        description="The normalization method to apply to gradients."
    )
    image_smoothing: bool = Field(
        default=False,
        description="If the perturbation should be created as a patch of the original domain of the images.",
    )

    normalization_strategy: Literal["perturbation", "result", "none"] = Field(default="None",
                                                                              description="Normalization strategy. Decide if project on epsilon-radius sphere only perturbation, adversarial result image or do nothing.")

    @field_validator("gradient_normalizer")
    def valid_gradient_normalizer(cls, v):
        if not isinstance(v, Normalization):
            raise ValueError("gradient_normalizer must be an instance of Normalization.")
        return v

class MomentumFGSMAttackConfig(_NumericalMethodsAttackConfig):
    loss: nn.Module = Field(
        default_factory=lambda: LossComposer(losses={"misclassification": {}}),
        description="The loss function to use for the attack.",
    )
    gradient_normalizer: Normalization = Field(
        default_factory=lambda: SignNormalization(),
        description="The normalization method to apply to gradients."
    )
    image_smoothing: bool = Field(
        default=False,
        description="If the perturbation should be created as a patch of the original domain of the images.",
    )

    optim_momentum: float = Field(default=0.7, ge=0.0, le=1.00, title="SGD Momentum", description="SGD Optimizer momentum.")

    normalization_strategy: Literal["perturbation", "result", "none"] = Field(default="None",
                                                                              description="Normalization strategy. Decide if project on epsilon-radius sphere only perturbation, adversarial result image or do nothing.")

    @field_validator("gradient_normalizer")
    def valid_gradient_normalizer(cls, v):
        if not isinstance(v, Normalization):
            raise ValueError("gradient_normalizer must be an instance of Normalization.")
        return v

class NaFGSMAttackConfig(_NumericalMethodsAttackConfig):
    loss: nn.Module = Field(
        default_factory=lambda: LossComposer(losses={"misclassification": {}}),
        description="The loss function to use for the attack.",
    )
    gradient_normalizer: Normalization = Field(
        default_factory=lambda: SignNormalization(),
        description="The normalization method to apply to gradients."
    )
    image_smoothing: bool = Field(
        default=False,
        description="If the perturbation should be created as a patch of the original domain of the images.",
    )

    optim_lr: float = Field(default=0.1, ge=1e-5, le=1.00, title="Learning Rate", description="SGD Optimizer learning rate.")

    optim_momentum: float = Field(default=0.7, ge=0.0, le=1.00, title="SGD Momentum", description="SGD Optimizer momentum.")

    optim_nesterov: bool = Field(default=True, title="Nesterov Momentum", description="SGD Nesterov momentum.")

    normalization_strategy: Literal["perturbation", "result", "none"] = Field(default="None",
                                                                              description="Normalization strategy. Decide if project on epsilon-radius sphere only perturbation, adversarial result image or do nothing.")

    @field_validator("gradient_normalizer")
    def valid_gradient_normalizer(cls, v):
        if not isinstance(v, Normalization):
            raise ValueError("gradient_normalizer must be an instance of Normalization.")
        return v


@EvasionAttackFactory.register(
    name="Fast Gradient Sign Method (FGSM)",
    description="A white-box adversarial attack exploiting model's gradient sign with respect to the input.",
    task={Task.Classification},
    type=AttackType.Digital,
    knowledge=Knowledge.White
)
class FGSMAttack(_NumericalMethodsAttack):
    r"""Implements a class of white-box adversarial attacks whose common factor is that we use the gradient's sign
    only to find the optimal perturbation. This class of methods is categorized under the term FGSMAttacks,
    due to the existence of subtle disparities. Herafter, we give a small documentation of known implementations:

    1. Let ``optimizer=torch.optim.SGD`` (with no momentum) and 1 step of
        ``max_iters=1`` it corresponds to the implementation of the original FGSM as
        devised in [1]_.
    2. Let ``optimizer=torch.optim.SGD`` (with ``momentum > 0``) and more than
        ``1`` step, i.e. ``max_iters>1`` it corresponds to the implementation of
        Momentum FGSM described in [2]_.
    3. Let ``optimizer=torch.optim.SGD`` (with ``nesterov=True`` and ``momentum
        > 0``) and more than ``1`` step, i.e. ``max_iters > 1``, it corresponds to
        the implementation of Nesterov Accelerated Gradient Adversarial Attack of
        [3]_.

    More than one configuration is available since we can choose better optimizers, e.g. ADAM or even L-BFGS.

    :param config: The configuration can be tweaked via changing the following parameters:
        1. ``optimizer`` which specifies a type of :class:`torch.optim.Optimizer`,
        2. ``optimizer_params`` a dictionary that specifies the parameters of the given ``optimizer``.
        3. ``scheduler`` which specifies a type of :class:`torch.optim.lr_scheduler.LRScheduler` as further optimization
            procedure to tweak iteratively the learning rate of the given ``optimizer``.
        4. ``scheduler_params`` a dictionary that specifies the parameters of the given ``scheduler``.
        5. ``max_iters`` the number of iteration procedure to compute.

        Note: The default configuration uses :class:`torch.optim.Adam` with
        learning rate `1e-3`.

    Example::

    Consider a batch of images :math:`(B, C, H, W)`, which we denote as ``data_input``, and a corresponding
    one-hot encoded labels ``target_label`` of shape :math:`(B, N_c)`. Let ``model`` be a :class:`AttackedModel`.
    Then, we can use the attack as follows

    >>> from nn_trust.attack import EvasionAttackFactory
    >>> cnf = EvasionAttackFactory.get_config("fgsm", model=model, task=Task.Classification, targeted=True, max_iters=20)
    >>> atk = EvasionAttackFactory.create_attack(config=cnf)
    >>> atk.generate(data_input, target_label)

    .. [1] Goodfellow, Ian J. et al. “Explaining and Harnessing Adversarial Examples.” CoRR abs/1412.6572 (2014)
    .. [2] Dong, Yinpeng et al. “Discovering Adversarial Examples with Momentum.” ArXiv abs/1710.06081 (2017)
    .. [3] Lin, Jiadong, Chuanbiao Song, Kun He, Liwei Wang and John E. Hopcroft. “Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks.”
            arXiv: abs/1908.06281 (2020)
    """

    CONFIG_T = FGSMAttackConfig

@EvasionAttackFactory.register(
    name="Momentum FGSM",
    description="Iteration based FGSM enhanced with SGD momentum",
    task={Task.Classification},
    type=AttackType.Digital,
    knowledge=Knowledge.White
)
class MomentumFGSMAttack(_NumericalMethodsAttack):
    CONFIG_T = MomentumFGSMAttackConfig

@EvasionAttackFactory.register(
    name="Nesterov Accelerated FGSM",
    description="Iteration based FGSM enhanced with SGD nesterov momentum",
    task={Task.Classification},
    type=AttackType.Digital,
    knowledge=Knowledge.White
)
class NaFGSMAttack(_NumericalMethodsAttack):
    CONFIG_T = NaFGSMAttackConfig