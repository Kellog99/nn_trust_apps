import torch
from pydantic import Field

from nn_trust import Task
from nn_trust.loss._loss import Loss, LossConfig
from nn_trust.loss.loss_factory import LossFactory


class L0LossConfig(LossConfig):
    toll: float = Field(
        default=0.5,
        description="Tolerance to accept for the new image.",
        gt=0.0,
        title="Tolerance"
    )
    normalise: bool = Field(
        default=True,
        description="Normalization flag for the output.",
        title="Normalise"
    )


@LossFactory.register(
    name="L0",
    description="This loss represents the L0 sub-norm which count the number of entries that are non-zero or less than a certain tolerance.",
    task={Task.Classification, Task.Detection}
)
class L0Loss(Loss):
    r"""
    An adversarial perturbation aims to be as small as possible.
    This loss represents the L0 subnorm which count the number of entries that are non-zero, i.e.
    .. math::
        f(x)=\sum_{i=1}^n\mathbb{1}_{x_i\ne 0}
    This loss will tell the number of pixels that are modified by the perturbation over a certain tollerance, :math:`\epsilon`.
    """

    CONFIG_T = L0LossConfig

    def forward(self, x_adv: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Compute of pixels that have changed from the original image.

        :param x_adv: a tensor of shape :math:`(B, d_1, \dots, d_K)`.
        :param x: a tensor of shape :math:`(B, d_1, \dots, d_K)`.

        :returns: the average pixel change ``x_f`` and ``x_o``.
        """

        dims = list(range(1, len(x_adv.size())))
        loss = ((x_adv - x).abs() - self.toll).relu().sign()
        loss = loss.sum(dim=dims).pow(2)

        if self.config.normalise:
            # With this flag is like taking the average number of changed pixels
            # This normalisation is important when the images have lot of pixels
            # avoids too many fluctuation on the loss
            loss /= x[0].shape.numel()

        # loss is a tensor of shape B x 1
        return self.reduce(loss)
