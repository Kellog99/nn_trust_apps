import torch
import torch.nn as nn
from pydantic import Field

from nn_trust import Task
from nn_trust.loss._loss import Loss, LossConfig
from nn_trust.loss.loss_factory import LossFactory


class RenyiDivLossConfig(LossConfig):
    alpha: float = Field(
        default=1.,
        description="Scaling power of the Rényi divergence loss.",
        title="Scaling power"
    )
    bound_max: float = Field(
        default=2.,
        description="Domain's upper bound.",
        title="Upper Bound"
    )


@LossFactory.register(
    name="RenyiDivLoss",
    description="This loss aims to measure the discrepancy between the model's expected codomain and its output.",
    task={Task.Classification, Task.Segmentation, Task.Detection}
)
class RenyiDivLoss(Loss):
    r"""
    Rényi divergence measures how different two probability distributions are
    as a generalized Kullback-Leibler divergence.
    It offers a family of measures parameterized by a real number :math:`\alpha \in (0, \infty)`,
    which allows for adjusting the sensitivity to differences between the distributions (higher
    values of :math:`\alpha` emphasize larger differences).
    Given two discrete distributions :math:`P = \{p_i\}_{i=1}^n`, :math:`Q = \{q_i\}_{i=1}^n`,
    it is defined as

    .. math::
        D_\alpha (P \| Q) = \frac{1}{\alpha-1}\log\left(\sum_{i=1}^n \frac{p_i^\alpha}{q_i^{\alpha-1}}\right)

    For more information on `Rényi Divergence <https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy#R%C3%A9nyi_divergence>`_

    Note that the inputs of the forward functions are probabilities and not in the log-space as for
    `torch.nn.KLDivLoss` format, before passing it you should normalize via softmax to obtain a probability distribution.

    Example::

    For particular values of :math:`\alpha` we have that the Rényi divergence has a nice closed form.
    In particular for :math:`\alpha=1/2` we have

    .. math::
        D_{1/2} (P \| Q) = -2 \log \sum_{i=1}^n \sqrt{p_i q_i}

    In this example we check that this property is preserved in the implementation:

    >>> a = torch.Tensor([0.1, 0.2, 0.7]).repeat(3, 1)
    >>> b = torch.Tensor([0.4, 0.3, 0.2]).repeat(3, 1)
    >>> ren_div = RenyiDivLoss(alpha=0.5, reduction="batchmean")
    >>> ren_div(a, b)
    tensor(0.3991)
    >>> -2 * (torch.sum((a * b).pow(0.5), dim=-1).log()).mean()
    tensor(0.3991)
    """

    CONFIG_T = RenyiDivLossConfig

    def forward(self, pred: torch.Tensor, target: torch.Tensor, *args) -> torch.Tensor:
        if self.config.alpha == 0:
            return self.reduce(pred[target > 0])

        elif self.config.alpha >= float("inf"):
            if self.config.reduction == "batchmean":
                max_val = torch.amax(target.div(pred), dim=-1)
                max_val.log_()
                return max_val
            else:
                max_val = torch.max(target.div(pred).view(-1))
                max_val.log_()
                return max_val
        elif self.config.alpha == 1:
            return nn.functional.kl_div(pred.log(), target, log_target=False, reduction=self.config.reduction)

        # Implements in non-limit case
        renyi_div = target * (target.div(pred)).pow(self.config.alpha - 1)

        renyi_div = self.reduce(renyi_div)
        renyi_div.log_()
        renyi_div.div_(self.config.alpha - 1)
        return renyi_div
