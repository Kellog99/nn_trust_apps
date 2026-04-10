from typing import cast

import torch
from pydantic import Field

from nn_trust import Knowledge, AttackType
from nn_trust.attack import EvasionAttackFactory
from nn_trust.attack._evasion import EvasionAttackConfig
from nn_trust.attack.evasion.decision._boundary import _BoundaryAttack, _BoundaryAttackConfig, _TensorBuffer
from nn_trust.core import Task


class SphereProjectionAttackConfig(_BoundaryAttackConfig):
    epsilon: float = Field(
        default=1e-2,
        description="Step size for reducing the distance.",
        gt=0.0,
        lt=1.0
    )


@EvasionAttackFactory.register(
    name="Boundary Sphere Attack",
    description="This attack provides an accept-reject strategy for generating a good adversarial perturbation.",
    task={Task.Classification},
    type=AttackType.Digital,
    knowledge=Knowledge.White
)
class SphereProjectionAttack(_BoundaryAttack):
    """A boundary decision attack based on the paper of [#Bren18] but with some modifications:
        * In this attack the projection is truly on the sphere and it is not an approximation
        * It is possible to use multiple kind of norm

        Let eta\sim N(0,1) then
        x_{adv, k+1}= x + (1-epsilon)*\|x_{adv,k}-x\|/\|x_{adv,k}-x+\eta\|(x_{adv,k}-x+\eta)

        If the term "1-\epsilon" does not exist, the adversarial input is the projection on the sphere of radius x_{adv,k}-x
        With that term, the distance with the original image is automatically reduced.

        With this kind of attack, it is not necessary to save the best perturbation because, by construction, x_adv
         will be the best in terms of distances

    References:
        .. [#Bren18] Wieland Brendel (*), Jonas Rauber (*), Matthias Bethge,
           "Decision-Based Adversarial Attacks: Reliable Attacks
           Against Black-Box Machine Learning Models",
           https://arxiv.org/abs/1712.04248
    """

    CONFIG_T = SphereProjectionAttackConfig

    def __init__(self, config: EvasionAttackConfig):
        super().__init__(config)
        self._config = cast(SphereProjectionAttackConfig, self._config)

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
        delta_norm = torch.linalg.vector_norm(delta, dim=list(range(1, delta.dim())), keepdim=True)

        ############# Sample the perturbation #############
        # delta' = d + eta
        delta_prime = delta + torch.randn_like(delta).to(self._config.device) / (x[0].numel() ** 0.5)
        delta_prime_norm = torch.linalg.vector_norm(delta_prime, dim=list(range(1, delta.dim())), keepdim=True)

        ## scaling to be at most as the previous distance
        delta_prime = (delta_norm / delta_prime_norm) * delta_prime
        ## reducing the distance of a factor (1-epsilon)
        delta_prime *= 1 - self._config.epsilon

        return x + delta_prime

    def update_parameters(self, buffer_parameters: _TensorBuffer, is_adversarial: torch.Tensor, **kwargs) -> None:
        if torch.logical_or(
                torch.all(torch.logical_not(is_adversarial)), buffer_parameters.get_differences() < self._config.toll
        ):
            # If all the candidates are not adversarial, reduce the step sizes
            self._config.epsilon *= 0.9
