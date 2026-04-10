import torch

from nn_trust import Task
from nn_trust.loss._loss import Loss, LossConfig
from nn_trust.loss.loss_factory import LossFactory


class TotalVariationLossConfig(LossConfig):
    pass


@LossFactory.register(
    name="Total Variation",
    description="This loss aims to measure the discrepancy between the model's expected codomain and its output.",
    task={Task.Classification, Task.Segmentation, Task.Detection}
)
class TotalVariationLoss(Loss):
    r"""
    Total variation loss in the Adversarial ML context helps to generate smoother images when attacking, avoiding
    pixelation that can be left out with camera noise and/or compression.

    The total variation comes from the approximation of
    .. math::
        \int_{\Omega} |\nabla f(x,y)|_2 d(x,y)
    In theory, $\Omega$ is the set of pixels, $P$, of an image, $p\in \R^{C \times W \times H}$. However, the approximation of of $\nabla f$ is given by

    .. math::
        \nabla f(x) \approx ((f(x+h,y)-f(x,y))/h, (f(x,y+h)-f(x,y))/h)

    Therefore the right corner, i.e. the pixels ${(i,W)}_{i=1}^H$, and the bottom corner, i.e. the pixels ${(W,i)}_{i=1}^H$, cannot be considered because in those points the derivative cannot be computed.
    Hence, let $P$ the set of pixels that allow the computation of the derivative, thus the discretization of the formula is given by

    .. math::
        L_{tv} = \sum_{(i, j)\in P} \sqrt{ \left(p_{i,j} - p_{i+1, j}\right)^{2} + \left(p_{i,j} - p_{i, j+1}\right)^{2}}

    Implementation from: Sharif, M., et al., "Accessorize to a Crime: Real and Stealthy Attacks on State-of-the-Art Face
    Recognition," in Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security, 2016,
    pp. 1528–1540.
    """

    CONFIG_T = TotalVariationLossConfig

    def forward(self, x_adv: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the total variation loss of a batch of images

        :param x_adv: the image batch.

        :return: the total variation loss value.
        """
        if x_adv.dim() != 4:
            AssertionError("Expected batched 4D tensor (batch x C X H x W).")
        tv_h = torch.pow(x_adv[..., :-1, :] - x_adv[..., 1:, :], self.config.p).sum(dim=(2, 3))
        tv_w = torch.pow(x_adv[..., :-1] - x_adv[..., 1:], self.config.p).sum(dim=(2, 3))
        res = (tv_h + tv_w).pow(1 / self.config.p)
        return self.reduce(res)
