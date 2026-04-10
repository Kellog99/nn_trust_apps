from typing import cast

import torch
from pydantic import Field

from nn_trust.attack import EvasionAttackFactory
from nn_trust.attack._evasion import EvasionAttackConfig
from nn_trust.attack.evasion.decision._boundary import _BoundaryAttack, _BoundaryAttackConfig, _TensorBuffer
from nn_trust.core import Task, AttackType, Knowledge


class BoundaryAttackConfig(_BoundaryAttackConfig):
    gamma: float = Field(
        default=0.04,
        description="Step size of the orthogonality direction.",
        gt=0.0,
        lt=1.0,
        title="Orthogonality direction step size"
    )

    beta: float = Field(
        default=0.07,
        description="Step size of the proximity direction.",
        gt=0.0,
        lt=1
    )

    epsilon: float = Field(
        default=0.3,
        description="Right hand side of the update.",
        gt=0.0,
        lt=1.0
    )

    fixed: bool = Field(
        default=False,
        description="Tells if the parameter beta is fixed or not."
    )


@EvasionAttackFactory.register(
    name="Boundary",
    description="A black-box adversarial attack that finds the optimal perturbation for each class and returns the most realistic modification.",
    task={Task.Classification, Task.Segmentation},
    type=AttackType.Digital,
    knowledge=Knowledge.White
)
class BoundaryAttack(_BoundaryAttack):
    """A powerful adversarial attack that requires neither gradients
    nor probabilities.

    This is the reference implementation for the attack. [#Bren18]_

    References:
        .. [#Bren18] Wieland Brendel (*), Jonas Rauber (*), Matthias Bethge,
           "Decision-Based Adversarial Attacks: Reliable Attacks
           Against Black-Box Machine Learning Models",
           https://arxiv.org/abs/1712.04248
    """

    CONFIG_T = BoundaryAttackConfig

    def __init__(self, config: EvasionAttackConfig):
        super().__init__(config)
        self._config = cast(BoundaryAttackConfig, self._config)

    def get_movement(self, x: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
        """
        The element that are passed here are those that can be updated.
        """
        if x.dim() == 3:
            # This if cover the case where only one element is passed inside this function
            # it is necessary to unsqueeze because this function works for batch
            x = x.unsqueeze(0)
            x_adv = x_adv.unsqueeze(0)

        delta = x_adv - x
        eta = torch.randn_like(delta).to(self._config.device) / (x[0].numel() ** 0.5)

        # orthogonal projection
        delta_norm = torch.linalg.vector_norm(delta, ord=self._config.p, dim=list(range(1, delta.dim())), keepdim=True)
        scalar_product = torch.sum(delta * eta, dim=list(range(1, delta.dim())), keepdim=True)
        eta_ortho = eta - (scalar_product / delta_norm.pow(self._config.p)) * delta

        # Normalization of the orthogonal perturbation
        eta_orth_norm = torch.linalg.vector_norm(
            eta_ortho, dim=list(range(1, delta.dim())), ord=self._config.p, keepdim=True
        ).clamp(self._config.toll)
        # eta = gamma * (\|delta\|/\|eta_ortho\|) * eta_ortho
        eta_ortho = self._config.gamma * (delta_norm / eta_orth_norm) * eta_ortho

        # Get the best value of beta
        beta = self._config.beta
        if not self._config.fixed:
            # in this case beta it is set with a proper value as in the theory
            beta = torch.tensor(self._config.beta, device=self._config.device)
            beta = beta.repeat(x.shape[0]).view(-1, *[1] * delta_norm[0].dim())
            beta.clamp(self._config.gamma * delta_norm, (self._config.gamma + self._config.epsilon) * delta_norm)
            self._config.beta = (
                    (self._config.gamma * delta_norm + (self._config.gamma + self._config.epsilon) * delta_norm) / 2
            ).mean()

        candidate = x_adv + self._config.gamma * eta_ortho - beta * delta / delta_norm

        return candidate

    def update_parameters(self, buffer_parameters: _TensorBuffer, is_adversarial: torch.Tensor, **kwargs) -> None:
        if torch.logical_or(
                torch.all(torch.logical_not(is_adversarial)), buffer_parameters.get_differences() < self._config.toll
        ):
            # If all the candidates are not adversarial, reduce the step sizes
            self._config.gamma *= 0.9
            self._config.beta *= 0.9

    def __repr__(self):
        return "Boundary Attack"
