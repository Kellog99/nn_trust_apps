import abc
import warnings
from collections.abc import Callable

import torch
import torch.nn.functional as F


class GradientEstimator(abc.ABC):
    r"""Gradient Estimator abstract class.

    Example::

    Let :class:`ConcreteGradientEstimator` be a concrete implementation of :class:`GradientEstimator`, then
    an example snippet is given by

    >>> x = torch.rand((1, 100), requires_grad=False)
    >>> y = (torch.rand(1, requires_grad=False) * 100).long()
    >>> grad_approx = ConcreteGradientEstimator()
    >>> x_grad_approx = grad_approx.gradient(lambda _x: torch.nn.CrossEntropyLoss()(_x, y), x)
    >>> x_grad_approx.shape
    torch.Size([1, 100])
    """

    @torch.no_grad()
    @abc.abstractmethod
    def gradient(self, loss: Callable[[torch.Tensor], torch.Tensor], params: torch.Tensor) -> torch.Tensor:
        r"""Approximates the gradient of the function ``loss`` with respect to the
        parameters ``params``.

        :param loss: a function :math:`\ell : \mathcal{X} \to \mathbb{R}` with
            :math:`\mathcal{X}` compatible with ``params``.
        :type loss: ``Callable[[torch.Tensor], torch.Tensor]``
        :param params: a tensor of parameters that is compatible with ``loss``'s input.
        :type params: ``torch.Tensor``

        :returns: A tensor with the same shape of ``params`` representing :math:`\nabla_x \ell(x)`.
        """

    def __call__(self, *args, **kwargs):
        return self.gradient(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}: " + str({k: self.__dict__[k] for k in self.__dict__})


class OrthogonalUnitaryCenteredDiffGradientApproximizer(GradientEstimator):
    r"""Estimates the gradient of a scalar function by using finite centered difference
    averaged on unitary orthogonal vectors directions. That is,

    .. math::
        \nabla_x f(x) \approx \frac{1}{N}\sum_{i=1}^N \frac{f(x + \varepsilon \cdot u_i) -
                    f(x - \varepsilon \cdot u_i)}{2\varepsilon^{\alpha}} u_i

    with :math:`N`,  :math:`\alpha` and :math:`\varepsilon` respectively corresponds to the arguments
    ``n_samples``, ``scaling`` and ``epsilon``.

    :param n_samples: number of samples used by the approximation method. Default is ``100``.
    :type n_samples: *int*
    :param epsilon: the maximum perturbation range used to sample the direction of the gradient. Default is ``1.0e-3``.
    :type epsilon: *float*
    :param scaling: the exponent of the scaling of the average normalization. The idea is that if the process is
        gaussian, then the scaling may be different from :math:`1`. Default is ``1.0``.
    :type scaling: *float*

    Example::

    We can check that the approximation is coherent with the backpropagation approximation of the gradient
    as exemplified in the following snippet

    >>> torch.manual_seed(131465)
    >>> x = torch.rand((1, 10), requires_grad=False, device="cpu")
    >>> y = (torch.rand(1, requires_grad=False, device="cpu") * 10).long()
    >>> oucda = OrthogonalUnitaryCenteredDiffGradientApproximizer(n_samples=8, epsilon=0.0001, scaling=0.5)
    >>> x_grad = oucda.gradient(loss=lambda _x: torch.nn.CrossEntropyLoss()(_x, y), params=x)
    >>> # We can compare the number of coordinates on which signs of the gradient is not correct
    >>> # by checking the approximation by backpropagation.
    >>> x.requires_grad_(True)
    >>> v = torch.nn.CrossEntropyLoss()(x, y)
    >>> v.backward()
    >>> torch.norm(x.grad.data.sign() - x_grad.sign(), p=1) / 2
    tensor(3.)

    """

    def __init__(self, n_samples: int = 100, epsilon: float = 1e-3, scaling: float = 1.0):
        self.n_samples = n_samples
        self.epsilon = epsilon
        self.scaling = scaling
        self.n_query = 2 * self.n_samples

    @staticmethod
    def _ortho_sphere_sample(n: int, shape: torch.Size, device: str | torch.device = "cpu") -> torch.Tensor:
        r"""Returns random orthogonal vectors on a unitary sphere.

        Note: This algorithm uses Graham-Schmidt orthogonalization procedure.
        This is quite computationally intensive as it is *O*\ (*d* *n*\ :sup:`2`) with *n* being the number
        of samples and *d* the output vector dimension.

        :param n: Number of sample vectors.
        :type n: int.
        :param shape: a shape representing the output of the tensor, hereafter
            denoted as :math:`(d_1, \dots, d_l)`.
        :type shape: :class:`torch.Size`.
        :param device: the device on which the computations should be executed. Default is 'cpu'.
        :type device: str.

        :returns: A tensor of shape :math:`(k, d_1, \dots, d_l)` with :math:`k=\min \{n, \prod_{i=1}^l d_i\}`
        """
        d = shape.numel()
        if d < n:
            warnings.warn(f"The number of samples n ({n=}) is larger than the vector space size ({d=}).", UserWarning)
        k = min(n, d)
        tensor_sample = torch.randn((k, d), device=device)
        # Normalize on the unit sphere in L^2
        tensor_sample[0] /= tensor_sample[0].norm(p=2.0).clamp_min(1e-12)
        for i in range(1, k):
            prev_basis = tensor_sample[:i]
            # Computes projection scaling for the vectors v_0,..., v_{i-1}
            scl_prods = torch.einsum("ij, j -> i", prev_basis, tensor_sample[i])
            # Computes the Graham-Schmidt orthogonalization the i-th vector
            tensor_sample[i] -= torch.einsum("j, jl -> l", scl_prods, prev_basis)
            # Normalize the i-th vector
            tensor_sample[i] /= tensor_sample[i].norm(p=2.0).clamp_min(1e-12)

        return tensor_sample.view((k, *list(shape)))

    def gradient(self, loss: Callable[[torch.Tensor], torch.Tensor], params: torch.Tensor) -> torch.Tensor:
        def centered_diffs(dt):
            return (loss(params + dt) - loss(params - dt)) * dt

        dir_samples = (
            OrthogonalUnitaryCenteredDiffGradientApproximizer._ortho_sphere_sample(
                self.n_samples, params.size(), device=params.device
            )
            * self.epsilon
        )
        est_grad = (torch.vmap(centered_diffs)(dir_samples) / (2 * self.epsilon**self.scaling)).sum(dim=0)
        return est_grad


class StochasticCoordinateDescentEstimator(GradientEstimator):
    r"""
    This is the method described in the offical zoo paper (https://doi.org/10.48550/arXiv.1708.03999).
    This method is related to coordinate descent method, which says that
    it is possible to estimate the gradient by taking finite differences on single coordinates.

    .. math:: \hat{g}_i \simeq \frac{\partial{f}}{\partial{x_i}}
    .. math:: \hat{g} \simeq \sum_i \frac{\partial{f}}{\partial{x_i}}

    We evaluate the gradient at batch_size coordinates, performing a total of
    2 * batch_size + 1 at each optimization round.

    :param n_samples: number coordinates to be estimated in a single iteration. Default is ``64``.
    :type n_samples: *int*
    :param epsilon: Step size used in computing the gradient. Default is ``1.0e-3``.
    :type epsilon: *float*

    Example:
    >>> input = torch.rand(size=(1, 3, 32, 32))
    >>> estimator = StochasticCoordinateDescentEstimator(n_samples=16)
    >>> g = estimator.gradient(loss=lambda x: x.sum(), params=input)
    >>> assert input.shape == g.shape

    """

    def __init__(self, n_samples: int = 64, epsilon: float = 1e-3):
        self.n_samples = n_samples
        self.epsilon = epsilon
        self.n_query = 2 * self.n_samples

    def _random_coordinate_direction(self, size: int):
        """Extract a set of random direction vector by
        Selecting a sample of self.n_samples coordinates without replacement
        then each generated vector has only one coordinates set to 1, and 0 elsewhere
        """
        coordinate_index_list = torch.randperm(size)[: self.n_samples]
        v = F.one_hot(coordinate_index_list, num_classes=size)
        return v

    def gradient(self, loss: Callable[[torch.Tensor], torch.Tensor], params: torch.Tensor):
        """
        :param loss: A function that compute the loss required for gradient estimation
        :type epsilon: *Callable*
        :param params: Tensor of paramters on which the gradient is required
        :type params: *torch.Tensor*
        """
        params_flat = params.view(-1)

        def centered_diffs(dt):
            return (loss(params + self.epsilon * dt) - loss(params - self.epsilon * dt)) * dt / (2 * self.epsilon)

        u = (
            self._random_coordinate_direction(size=params_flat.numel())
            .view(self.n_samples, *params.size())
            .to(params.device)
        )
        res = torch.vmap(centered_diffs)(u).sum(0)
        return res


class RandomCoordinateEstimator(GradientEstimator):
    """
    Randomly select a subset of possible coordinates in order to create a direction
    that is used to compute the gradient.

    :param n_samples: number coordinates to be selected randomly when deciding the direction on which the gradient has to be estimated. Default is ``64``.
    :type n_samples: *int*
    :param epsilon: Step size used in computing the gradient. Default is ``1.0e-3``.
    :type epsilon: *float*

    Example:
    >>> input = torch.rand(size=(1, 3, 32, 32))
    >>> estimator = RandomCoordinateEstimator(n_samples=16)
    >>> g = estimator.gradient(loss=lambda x: x.sum(), params=input)
    >>> assert input.shape == g.shape
    """

    def __init__(self, n_samples: int = 100, epsilon: float = 1e-3):
        self.n_samples = n_samples
        self.epsilon = epsilon
        self.n_query = 2

    def _random_coordinate_direction(self, size: int):
        """Generate a single random direction vector
        by setting to 1 only a set on self.n_samples coordinates
        """
        coordinate_index_list = torch.randperm(size)[: self.n_samples]
        v = torch.zeros(size=(size,))
        v[coordinate_index_list] = 1
        return v

    def gradient(self, loss: Callable[[torch.Tensor], torch.Tensor], params: torch.Tensor):
        """
        :param loss: A function that compute the loss required for gradient estimation
        :type epsilon: *Callable*
        :param params: Tensor of paramters on which the gradient is required
        :type params: *torch.Tensor*
        """
        params_flat = params.view(-1)

        def centered_diffs(dt):
            return (loss(params + self.epsilon * dt) - loss(params - self.epsilon * dt)) * dt / (2 * self.epsilon)

        u = self._random_coordinate_direction(size=params_flat.numel()).view(*params.size()).to(params.device)
        res = torch.vmap(centered_diffs)(u)
        return res


class RandomUnitDirectionEstimator(GradientEstimator):
    """
    This method extract a set of n_samples random direction on the unit hypersphere.
    These directions are drawn from a random uniform distribution.

    Example:
    >>> input = torch.rand(size=(1, 3, 32, 32))
    >>> estimator = RandomUnitDirectionEstimator(n_samples=1)
    >>> g = estimator.gradient(loss=lambda x: x.sum(), params=input)
    >>> assert input.shape == g.shape

    """

    def __init__(
        self,
        n_samples: int = 1,
        epsilon: float = 1e-3,
    ):
        self.n_samples = n_samples
        self.epsilon = epsilon
        self.n_query = 2 * self.n_samples

    def _random_unit_vector(self, size: int):
        """Extract a batch of vectors (B, N) on the unit hypersphere"""
        v = torch.rand(size=(self.n_samples, size))
        u = v / v.norm(p=2.0, dim=1).view(self.n_samples, -1)
        return u

    def gradient(self, loss: Callable[[torch.Tensor], torch.Tensor], params: torch.Tensor):
        """
        :param loss: A function that compute the loss required for gradient estimation
        :type epsilon: *Callable*
        :param params: Tensor of paramters on which the gradient is required
        :type params: *torch.Tensor*
        """
        params_flat = params.view(-1)

        def centered_diffs(dt):
            return (loss(params + self.epsilon * dt) - loss(params - self.epsilon * dt)) * dt / (2 * self.epsilon)

        direction_vecs = (
            self._random_unit_vector(size=params_flat.numel()).view(self.n_samples, *params.size()).to(params.device)
        )
        grad_estimations = torch.vmap(centered_diffs)(direction_vecs)
        return grad_estimations.mean(dim=0)
