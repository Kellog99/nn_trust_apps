from typing import Self

import torch
from pydantic import Field, model_validator

from nn_trust import Task
from nn_trust.loss._loss import Loss, LossConfig
from nn_trust.loss.loss_factory import LossFactory


class BoundLossConfig(LossConfig):
    bound_min: float = Field(
        default=-2.,
        description="Domain's lower bound.",
        title="Lower Bound"
    )
    bound_max: float = Field(
        default=2.,
        description="Domain's upper bound.",
        title="Upper Bound"
    )

    @model_validator(mode='after')
    def check_bounds_order(self) -> Self:
        """
        Ensures that bound_max is strictly greater than bound_min.
        """
        # The 'after' mode receives the partially built model instance 'self'
        c_min = self.bound_min
        c_max = self.bound_max

        # Check if bound_max is strictly larger than bound_min
        if c_max <= c_min:
            # Raise a ValueError with a helpful message
            raise ValueError(
                f"c_max must be larger than c_min instead it received "
                f"(c_max, c_min) = ({c_max}, {c_min})."
            )
        return self


@LossFactory.register(
    name="Bound",
    description="This loss aims to measure the discrepancy between the model's expected codomain and its output.",
    task={Task.Classification, Task.Segmentation, Task.Detection}
)
class BoundLoss(Loss):
    r"""
    Let :math:`C \subset \mathbb{R}^n`, and :math:`x \in \mathbb{R}^n`, then this function
    computes the distance between :math:`x` and a target set :math:`C`. In particular,
    :math:`C` is an :math:`n`-dimensional box of specified width, i.e. :math:`C = [c_{min}, c_{max}]^n`.
    The loss may be described in a closed form as:

    .. math::
        \begin{align*}
        f:\mathbb{R}^n \to\mathbb{R}\\
            x&\mapsto f(x)\coloneqq
            \begin{cases}
                0  & x\in C \\
                dist_p(x, C) & \text{otherwise}
            \end{cases}
        \end{align*}
    """
    CONFIG_T = BoundLossConfig

    def forward(self, x_adv: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Computes the bounding loss of a batched tensor ``x``.

        :param x_adv: a tensor of shape :math:`(B, d_1, \dots, d_K)`.

        :returns: the bounding loss with respect to the axis :math:`d_1, \dots, d_K`.
        :returns: the bounding loss with respect to the axis :math:`d_1, \dots, d_K`.
        """

        dims = list(range(1, len(x_adv.size())))

        upper = torch.sum(torch.relu(x_adv - self.config.bound_max).pow(self.config.p), dim=dims)
        lower = torch.sum(torch.relu(self.config.bound_min - x_adv).pow(self.config.p), dim=dims)
        return self.reduce(upper + lower)
