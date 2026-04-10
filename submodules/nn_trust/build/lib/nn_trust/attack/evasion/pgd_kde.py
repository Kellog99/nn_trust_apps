from typing import Literal, Optional

import torch
import torch.nn as nn
from pydantic import Field

from nn_trust.attack import EvasionAttackFactory
from nn_trust.attack.evasion._numerical import _NumericalMethodsAttack, _NumericalMethodsAttackConfig
from nn_trust.attack.normalization import LpNormalization, Normalization
from nn_trust.attack.utils._utils import to_device
from nn_trust.core import AttackType, Knowledge, Task
from nn_trust.loss import MisclassificationLoss
from nn_trust.loss.losses.bregmandiv import BregmanDivLoss, BregmanDivLossConfig
from nn_trust.loss.losses.misclassification import MisclassificationLossConfig


class PGDKDEAttackConfig(_NumericalMethodsAttackConfig):
    loss: nn.Module = Field(
        default_factory=lambda: BregmanDivLoss(BregmanDivLossConfig()),
        description="Loss function to use for the attack.",
    )
    maximum_gaussian_kernel_density_points: int = Field(
        default=20,
        description="Maximum number of sample points used for the gaussian density function.",
        gt=0,
        title="Maximum number of sample points"
    )
    regularization_kernel_density: float = Field(
        default=0.5,
        description="The influence of the kernel density estimation function.",
        gt=0.0,
        title="Kernel influence"
    )
    normalizer: Normalization = Field(
        default_factory=lambda: LpNormalization(p=2.0, radius=20.0),
        description="Gradient normalizer to use during the attack.",
    )


@EvasionAttackFactory.register(
    name="Projected Gradient Descent with Kernel Density Estimation (PGDKDE)",
    description="White-box attack uses Projected Gradient Descent on an Lp sphere, leveraging data knowledge from Gaussian kernel density estimation.",
    task={Task.Classification, Task.Segmentation, Task.Detection},
    type=AttackType.Digital,
    knowledge=Knowledge.Black
)
class PGDKDEAttack(_NumericalMethodsAttack):
    r"""Implements a white-box adversarial attacks based on Projected Gradient Descent on an
    :math:`L^p` sphere. If additional information is passed via a training dataset, the method uses
    a gaussian kernel density estimate to infer additional properties on the target objective as described in
    [1]_. The loss used when a training dataset is provided is the following:

    .. math::
        L(x, y) = \textrm{CrossEntropy}(g(x), y) + \frac{\lambda}{n} \sum_{i \mid y = g(x_i)} k(x - x_i)

    with :math:`g(x)` being the evaluation of the model, :math:`k` is the gaussian kernel density estimation with respect to points
    :math:`x_i` that are classified analogously to the given target label :math:`y`.

    :param config: The configuration can be tweaked via changing the following parameters:
        1. ``optimizer`` which specifies a type of :class:`torch.optim.Optimizer`,
        2. ``optimizer_params`` a dictionary that specifies the parameters of the given ``optimizer``.
        3. ``scheduler`` which specifies a type of :class:`torch.optim.lr_scheduler.LRScheduler` as further optimization
            procedure to tweak iteratively the learning rate of the given ``optimizer``.
        4. ``scheduler_params`` a dictionary that specifies the parameters of the given ``scheduler``.
        5. ``max_iters`` the number of iteration procedure to compute.
        6. ``normalizer.p`` specifies the norm in the :math:`L^p` space.
        7. ``normalizer.radius`` specifies the maximum allowed norm in the given ``L^p`` space.
        8. ``maximum_gaussian_kernel_density`` specifies the maximum number of points used for the gaussian kernel density estimation.
        9. ``regularization_kernel_density`` specifies the weight the algorithm gives to the gaussian kernel density estimation.

    Example::

    Consider a batch of images :math:`(B, C, H, W)`, which we denote as ``data_input``, and a corresponding
    one-hot encoded labels ``target_label`` of shape :math:`(B, N_c)`. Let ``model`` be a :class:`AttackedModel`.
    Then, we can compute the Projected Gradient Descent (PGD) attack with respect to the :math:`L^\infty`-norm
    with radius :math:`0.03` as:

    >>> from nn_trust.attack import EvasionAttackFactory
    >>> cnf = EvasionAttackFactory.get_config("pgdkde", model=model, max_iters=20)
    >>> cnf.gradient_normalizer.p = float("inf")
    >>> cnf.gradient_normalizer.radius = 0.03
    >>> atk = EvasionAttackFactory.create(config=cnf)
    >>> atk.generate(data_input, target_label)

    .. [1] Biggio, Battista, Igino Corona, Davide Maiorca, Blaine Nelson, Nedim Srndic, Pavel Laskov, Giorgio Giacinto and Fabio Roli. “Evasion Attacks against Machine Learning at Test Time.” ArXiv abs/1708.06131 (2013)
    """

    CONFIG_T = PGDKDEAttackConfig

    def fit(
            self,
            proxy_data: torch.utils.data.DataLoader,
            **kwargs,
    ) -> None:
        to_device(self._config, self._config.device)
        # If a training dataloader is found, we collect the training samples to create a gaussian kernel density estimator
        # to use as a metric distance between the sample to test and the training distribution.
        sample_points = None
        sample_classification_points = None
        j = 0
        for x, y in proxy_data:
            # Initialize the sample points used for Gaussian Density Kernel Estimation
            if sample_points is None:
                sample_points = torch.zeros(
                    (self._config.maximum_gaussian_kernel_density_points, *x.shape[1:]),
                    dtype=x.dtype,
                    device=x.device,
                )
                sample_classification_points = torch.zeros(
                    self._config.maximum_gaussian_kernel_density_points,
                    dtype=y.dtype,
                    device=y.device
                )

            max_j = min(j + x.shape[0], self._config.maximum_gaussian_kernel_density_points)
            # Note the ys should be integers representing the class, not a one-hot encoding vector
            if y.dim() > 1:
                y = y.abs().argmax(dim=-1)
            sample_points[j:max_j] = x.clone()[: max_j - j]
            sample_classification_points[j:max_j] = y.clone()[: max_j - j]
            j += x.shape[0]
            # Breaks
            if max_j == self._config.maximum_gaussian_kernel_density_points:
                break

        # Augment the Misclassification Loss with the Gaussian Kernel Density Function
        self._config.loss = ReduceModule(
            modules=[
                GaussianKernelDensityFunction(
                    samples=sample_points,
                    classification_samples=sample_classification_points
                ),
                MisclassificationLoss(MisclassificationLossConfig(losstype=0)),
            ],
            weights=[self._config.regularization_kernel_density, 1.0],
        )

    def reset_fit(self):
        self._config.loss = MisclassificationLoss(MisclassificationLossConfig(losstype=0))


class GaussianKernelDensityFunction(nn.Module):
    r"""Computes the Gaussian Kernel estimate of a new sample with respect to a given sample. That is, suppose a point :math:`x`
    is classified in the cluster :math:`i`, and the sample is given by a collection of points with their corresponding
    'classification cluster', e.g. :math:`\{ (x_i, y_i)\}` with :math:`y_i \in \{0, \dots, N\}`, then
    .. math::
        k(x) = \sum_{j \mid y_j = i} \exp\left(-\frac{\| x - x_j \|^2}{h}\right)

    :param samples: A tensor of :math:`(B, d_1, \dots, d_k)` shape, corresponding to ``B`` examples.
    :param classification_samples: A tensor of :math:`(B)` shape, corresponding to the classification values of the
        ``samples`` inputs.
    :param bandwidth: a float values that scales the Gaussian kernel density.
    :param normalization: Either takes the mean or the sum of all its kernel estimates. Default is 'mean'.
    """

    def __init__(
            self,
            samples: torch.Tensor,
            classification_samples: torch.Tensor,
            bandwidth: float = 10,
            normalization: Literal["sum", "mean"] = "mean",
    ):
        super().__init__()
        self._bandwidth = bandwidth
        self._samples = samples
        self._classification_samples = classification_samples
        self._N = self._samples.shape[0]
        if self._classification_samples.shape[0] != self._N:
            raise ValueError("The given 'samples' do not have the same size of the 'classification_samples'.")
        self._normalization = normalization

    def forward(
            self,
            x_adv: torch.Tensor,
            target: torch.Tensor,
            *args,
            **kwargs
    ) -> torch.Tensor:
        r"""Computes the Gaussian Kernel estimate given the correct evaluation of the input ``x`` as ``y``.

        :param x_adv: a batch of tensors :math:`(B, d_1, \dots, d_k)`
        :param target: a batch of tensors :math:`(B)` classification of the given inputs ``x``

        :returns: The result of the computation.
        """
        if self._classification_samples is None or self._samples is None:
            raise RuntimeError("The classification samples used for the estimate are not defined.")

        # If the targets are in one-hot encoding, pass to the token representation
        if target.dim() == 2:
            target = target.argmax(dim=-1)

        good_indexes = torch.stack([self._classification_samples == yi.item() for yi in target])
        kernels = torch.zeros(x_adv.shape[0], device=x_adv.device, dtype=x_adv.dtype)
        for b in range(x_adv.shape[0]):
            n_good_indices = good_indexes[b].int().sum().item()
            if n_good_indices < 1:
                continue
            pkernels = (
                (self._samples[good_indexes[b]].to(x_adv.device) - x_adv[b])
                .view(n_good_indices, -1)
                .pow(2)
                .sum(dim=-1)
                .neg()
                .div(self._bandwidth)
            )
            if self._normalization == "mean":
                kernels[b] = pkernels.mean()
            elif self._normalization == "sum":
                kernels[b] = pkernels.sum()
            else:
                raise NotImplementedError(f"The given normalization '{self._normalization}' is not implemented.")
        return kernels.mean()


class ReduceModule(nn.Module):
    r"""A module that composes the input modules by adding their results together.

    :param modules: a list of :class:`torch.nn.Module`.
    """

    def __init__(self, modules: list[nn.Module], weights: Optional[list[float]]):
        super().__init__()
        if weights is None:
            weights = [1.0] * len(modules)

        if len(modules) != len(weights):
            raise ValueError("The length of weights and modules must be the same.")

        self._all_modules = modules
        self._weights = weights

    def forward(self, *args, **kwargs):
        r"""Computes the summation of the output of the given modules with respect to
        the given inputs.
        """
        res = None
        for weight, mod in zip(self._weights, self._all_modules, strict=False):
            if res is None:
                res = weight * mod(*args, **kwargs)
            else:
                res += weight * mod(*args, **kwargs)

        return res
