from collections.abc import Callable

import torch
from pydantic import Field

from nn_trust import Task
from nn_trust.loss._loss import Loss, LossConfig
from nn_trust.loss.loss_factory import LossFactory


class BregmanDivLossConfig(LossConfig):
    phi: Callable[[torch.Tensor], torch.Tensor] = Field(
        default=torch.nn.functional.sigmoid,
        description="A convex function that takes a batch :math:`(B, d_1, \dots, d_k)` and returns :math:`(B, 1)`",
        title="Phi function"
    )


@LossFactory.register(
    name="Bregman Divergence",
    description="It measures the dissimilarity between two points in a convex set described by a convex function.",
    task={Task.Classification, Task.Segmentation, Task.Detection}
)
class BregmanDivLoss(Loss):
    r"""
    Bregman divergence measures the dissimilarity between two points in a convex set described
    by a convex function :math:`\phi`. It generalizes the squared Euclidean distance and the Kullback-Leibler divergence.
    It is computed by taking the difference between the function evaluated at one point and the sum of the function
    evaluated at the other point and the gradient of the function evaluated at the first point.
    For two points :math:`x,y` and a given convex function :math:`\phi`, it is defined as
    .. math::
            D_\text{Bregman}(x,y) \coloneqq \phi(x) - \phi(y) - \langle \nabla \phi(y), x-y\rangle.

    For more information on `Bregman Divergence <https://en.wikipedia.org/wiki/Bregman_divergence>`_

    :heading level: 3
    Examples

    The Bregman Divergence Loss is the same as the MSE Loss as long as we use :math:`phi = \|x\|_2^2`. Namely,

    .. code-block:: python
    >>> torch.manual_seed(12314)
    >>> a = torch.randn(3, 20) * 10 + 2
    >>> b = torch.randn(3, 20) * 10 - 1
    >>> def phi_mse(x):
    >>>     return torch.norm(x, 2, dim=-1)**2.0
    >>> breg_div = BregmanDivLoss(phi_mse)
    >>> breg_div(a, b)
    tensor([4050.0693, 5120.1094, 7150.4912])
    >>> torch.norm(a - b, 2, dim=-1) ** 2.0
    tensor([4050.0688, 5120.1094, 7150.4912])

    Similarly, one can test it for the logistic regression loss by using :math:`phi = p\log(p) + (1-p)\log(1-p)`. Namely,

    >>> torch.manual_seed(1234)
    >>> targets = torch.randn(3, 1).relu().sign()
    >>> predicted_prob = torch.rand(3, 1)
    >>> def phi_logistic(p):
    >>>     p.data = torch.clamp(p, 0.00001, 0.999999)
    >>>     return (p * p.log() + (1-p) * (1-p).log()).sum(dim=-1)
    >>> breg_div = BregmanDivLoss(phi_logistic)
    >>> breg_div(targets, predicted_prob)
    tensor([0.3948, 1.1042, 1.5310])
    >>> -(targets * predicted_prob.log() + (1 - targets) * (1 - predicted_prob).log()).T
    tensor([[0.3948, 1.1042, 1.5311]])

    """

    CONFIG_T = BregmanDivLossConfig

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            *args
    ) -> torch.Tensor:
        r"""
        Denoting with :math:`\text{pred} = x` and :math:`\text{target} = y`, it returns

        .. math::
            \phi(x) - \phi(y) - \langle \nabla \phi(y), x-y\rangle.

        :param pred: A tensor of shape :math:`(B, d_1, \dots, d_k)` of same size as `target`.
        :type pred: torch.Tensor
        :param target: A tensor of shape :math:`(B, d_1, \dots, d_k)` of same size as `pred`.
            It must be also compatible with the function `phi` defined in the given instance
            of `BregmanDivergence`.
        :type target: torch.Tensor
        """
        # Computes the gradient of the target with respect to the given convex function _phi
        diff_target = target.clone().detach()
        diff_target.requires_grad_()
        der_phi = self.config.phi(diff_target)
        v = torch.ones(pred.shape[0])
        der_phi.backward(gradient=v)
        # Computes the inner product and finally returns
        # phi(pred) - phi(target) - <grad phi(target), pred-target>
        inner_prod = diff_target.grad.data * (pred - target)
        all_dims_except_batch_dim = list(range(1, len(inner_prod.shape)))
        return self.config.phi(pred) - self.config.phi(target) - torch.sum(inner_prod, dim=all_dims_except_batch_dim)
