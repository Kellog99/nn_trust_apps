import abc
import warnings
from typing import Optional

import torch


class Sampler(abc.ABC):
    r"""Implements a specialized Iterator for sampling :class:`torch.Tensors`. This allows for simple for-loop
    construction in the sampling architecture. Moreover, it defines a ``batch_size`` to allow for more efficient
    implementation, if available, otherwise defaults to ``batch_size = 1``.

    :param shape: a :class:`torch.Size` for the corresponding sampler output.
    :param batch_size: the number of sample that are generated each time ``__next__`` is called.
    :param device: the device operations are performed on. Default is ``'cpu'``.
    :param generator: the pseudo-random generator :class:`torch.Generator` to specify. If None, a new
        :class:`torch.Generator` is created on the specified device.
    """

    def __init__(
        self,
        shape: torch.Size,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
        generator: Optional[torch.Generator] = None,
    ):
        if batch_size < 1:
            raise ValueError("`batch_size` must be positive.")

        self._shape = shape
        self._batch_size = batch_size
        self._device = device
        if generator is None:
            self._generator = generator
        else:
            self._generator = torch.Generator(device=self._device)

    @abc.abstractmethod
    def sample(self, n: int) -> torch.Tensor:
        r"""Sample according to a specified technique a :class:`torch.Tensor` with shape :math:`(n, d_1, \dots, d_K)`
        with :math:`d_1, \dots, d_K` being the :class:`torch.Size` specified as argument of the initializer of the
        :class:`Sampler`.

        This method is called every time __next__ is called with the specified ``batch_size``
        size as ``n``.

        If the Sampler is a Markovian process, remember to update the state of the sampler to
        allow the ``__next__`` method to run consistently.

        :param n: number of :class:`torch.Tensor` to be sampled simultaneously.

        :returns: a :class:`torch.Tensor` with shape :math:`(n, d_1, \dots, d_K)`
            with :math:`d_1, \dots, d_K` being the :class:`torch.Size` specified as argument of the initializer of the
            :class:`Sampler`.
        """

    def __next__(self):
        res = self.sample(self._batch_size)
        return res

    def __iter__(self):
        returned_stop_iter = False
        while not returned_stop_iter:
            try:
                yield self.__next__()
            except StopIteration:
                returned_stop_iter = True


class UnitaryOrthoSampler(Sampler):
    r"""Samples orthogonal vectors on a unitary sphere.

    Note: This algorithm uses Graham-Schmidt orthogonalization procedure.
    This is quite computationally intensive as it is *O*\ (*d* *n*\ :sup:`2`) with *n* being the number
    of samples and *d* the output vector dimension.

    :param batch_size: Number of vectors to samples for each batch.
    :type batch_size: int.
    :param shape: a :class:`torch.Size` representing the output of the tensor, hereafter
        denoted as :math:`(d_1, \dots, d_K)`.
    :param device: the device on which the computations should be executed. Default is ``'cpu'``.
    :param generator: the pseudo-random generator :class:`torch.Generator` to specify. If None, a new
        :class:`torch.Generator` is created on the specified device.
    """

    def __init__(
        self,
        shape: torch.Size,
        batch_size: int = 1,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__(shape, batch_size, device, generator)
        self._previous_states = torch.zeros((0, int(self._shape.numel())), device=self._device)

    def sample(self, n: int) -> torch.Tensor:
        d = int(self._shape.numel())
        if d < n + self._previous_states.size(0):
            raise StopIteration("The orthonormal basis is complete. Further iterations are pointless.")

        new_basis = torch.rand(n, d, generator=self._generator, device=self._device)

        self._previous_states = _unitary_orthogonalization(
            basis_vectors=self._previous_states, new_vectors=new_basis, device=self._device
        )
        return self._previous_states[-n:].view(n, *self._shape)


class CartesianSampler(Sampler):
    r"""Samples orthogonal vectors on the standard Cartesian Orthonormal basis.

    :param batch_size: Number of vectors to samples for each batch.
    :type batch_size: int.
    :param shape: a :class:`torch.Size` representing the output of the tensor, hereafter
        denoted as :math:`(d_1, \dots, d_K)`.
    :param device: the device on which the computations should be executed. Default is ``'cpu'``.
    :param generator: the pseudo-random generator :class:`torch.Generator` to specify. If None, a new
        :class:`torch.Generator` is created on the specified device.
    """

    def __init__(
        self,
        shape: torch.Size,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__(shape, batch_size, device, generator)

    def sample(self, n: int) -> torch.Tensor:
        d = int(self._shape.numel())
        dir_indices = torch.randperm(d, generator=self._generator, device=self._device)[:n]
        batch_indices = torch.arange(n)
        sample = torch.zeros((n, d), device=self._device)
        sample[batch_indices, dir_indices] = 1.0

        return sample.view(n, *self._shape)


class DCTSampler(Sampler):
    r"""Samples vectors on the Discrete Cosine basis.

    :param batch_size: Number of vectors to samples for each batch.
    :type batch_size: int.
    :param shape: a :class:`torch.Size` representing the output of the tensor, hereafter
        denoted as :math:`(d_1, \dots, d_K)`.
    :param device: the device on which the computations should be executed. Default is ``'cpu'``.
    :param generator: the pseudo-random generator :class:`torch.Generator` to specify. If None, a new
        :class:`torch.Generator` is created on the specified device.
    """

    def __init__(
        self,
        shape: torch.Size,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
        min_freq: int = 0,
        max_freq: int = -1,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__(shape, batch_size, device, generator)
        self._min_freq = min_freq
        self._max_freq = max_freq

    def sample(self, n: int) -> torch.Tensor:
        d = int(self._shape.numel())
        max_freq = d if self._max_freq == -1 else self._max_freq
        # Generate the Cartesian coordinates
        dir_indices = torch.randperm(d, generator=self._generator, device=self._device)
        dir_indices = dir_indices[dir_indices >= self._min_freq]
        dir_indices = dir_indices[dir_indices <= max_freq][:n]
        batch_indices = torch.arange(n)
        basis = torch.zeros((n, d), device=self._device)
        basis[batch_indices, dir_indices] = 1.0
        # Pass to the discrete Cosine basis
        k = -torch.arange(d, device=self._device) * torch.pi / (2 * d)
        Wr = torch.cos(k)
        Wi = torch.sin(k)
        coeffs = torch.view_as_real(torch.fft.fft(basis, dim=1))
        V = coeffs[..., 0] * Wr + coeffs[..., 1] * Wi
        # normalize
        V[:, 0] /= d**0.5
        V[:, 1:] /= (d / 2) ** 0.5
        return V.view(n, *self._shape)


class BallSampler(Sampler):
    r"""Samples vectors within a ball of specified radius.

    :param shape: a :class:`torch.Size` for the corresponding sampler output.
    :param batch_size: the number of sample that are generated each time ``__next__`` is called.
    :param device: the device operations are performed on. Default is ``'cpu'``.
    :param generator: the pseudo-random generator :class:`torch.Generator` to specify. If None, a new
        :class:`torch.Generator` is created on the specified device.
    :param radius: specifies the radius of the ball to sample within.
    """

    def __init__(
        self,
        shape: torch.Size = None,
        radius: float = 1.0,
        batch_size: int = 1,
        version: int = 2,
        p: float = 2.0,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__(shape, batch_size, device, generator)
        self.p = p
        if self._shape is not None:
            self.setShape(shape)
        self._version = version
        self._radius = radius

    def setShape(self, shape: torch.Size):
        self._shape = shape
        self._dim = shape.numel()

    def sample(self, n: int) -> torch.Tensor:
        r"""Sample ``n`` vectors uniformly inside a ball of specified ``radius``.

        :param n: number of :class:`torch.Tensor` to be sampled.

        :returns: a :class:`torch.Tensor` with shape :math:`(n, d_1, \dots, d_K)`
        with :math:`d_1, \dots, d_K` being the :class:`torch.Size` specified as argument of the initializer of the
        :class:`Sampler`.
        """
        if self._shape is None:
            raise ValueError("The shape must be set to something")
        if self._generator is None:
            self._generator = torch.Generator(device=self._device)

        Z = torch.randn(n, self._dim, device=self._device, generator=self._generator)
        sphere = Z / torch.linalg.norm(Z, ord=2, dim=1, keepdim=True).clamp(1e-12)
        u = (1 / self._dim * torch.rand(n, 1, device=self._device, generator=self._generator).clamp(1e-12).log()).exp()
        radius = self._radius * u
        out = (radius * sphere).view(n, *self._shape)
        return out


def _unitary_orthogonalization(
    basis_vectors: torch.Tensor, new_vectors: torch.Tensor, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    r"""Given ``basis_vectors`` and ``new_vectors`` orthogonalize and normalize the latter
    with respect to the former. It assumes that the ``basis_vectors`` are an orthonormal basis
    already.

    :param basis_vectors: a :class:`torch.Tensor` with shape :math:`(n_1, d)`.
    :param new_vectors: a :class:`torch.Tensor` with shape :math:`(n_2, d)`.
    :param device: a :class:`torch.Device` specifies the device where the operations are performed.

    :returns: a new :class:`torch.Tensor` with shape :math:`(n_1 + n_2, d)` such that each row vector is
        unitary in :math:`L^2` norm and each couple of row has inner product :math:`0`, i.e. are orthogonal.
    """
    if basis_vectors.size(1) != new_vectors.size(1):
        raise ValueError("The vectors must have the same dimensions.")

    # Construct the new tensor containing all vectors
    all_vectors = torch.cat([basis_vectors, new_vectors])

    # Actually perform orthogonalization
    for i in range(basis_vectors.size(0), all_vectors.size(0)):
        prev_basis = all_vectors[:i]
        # Computes projection scaling for the vectors v_0,..., v_{i-1}
        scl_prods = torch.einsum("ij, j -> i", prev_basis, all_vectors[i])
        # Computes the Graham-Schmidt orthogonalization the i-th vector
        all_vectors[i] -= torch.einsum("j, jl -> l", scl_prods, prev_basis)
        # Normalize the i-th vector
        all_vectors[i] /= torch.norm(all_vectors[i]).clamp_min(1.0e-12)

    return all_vectors


def ortho_sphere_sample(n: int, shape: torch.Size, device: str = "cpu") -> torch.Tensor:
    r"""Returns random orthogonal vectors on a unitary sphere.

    .. Note:: this is the functional version of the sampling technique implemented in :class:`UnitaryOrthoSampler`.

    .. Note:: This algorithm uses Graham-Schmidt orthogonalization procedure.
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
    tensor_sample[0] /= torch.norm(tensor_sample[0])
    for i in range(1, k):
        prev_basis = tensor_sample[:i]
        # Computes projection scaling for the vectors v_0,..., v_{i-1}
        scl_prods = torch.einsum("ij, j -> i", prev_basis, tensor_sample[i])
        # Computes the Graham-Schmidt orthogonalization the i-th vector
        tensor_sample[i] -= torch.einsum("j, jl -> l", scl_prods, prev_basis)
        # Normalize the i-th vector
        tensor_sample[i] /= torch.norm(tensor_sample[i])

    return tensor_sample.view((k, *list(shape)))


def ball_sampling(
    dim: int,
    samples: int,
    radius: float = 2.5,
    device: torch.device = torch.device("cpu"),
    generator: Optional[torch.Generator] = None,
):
    r"""Computes a sample of size ``samples`` inside a sphere of specified ``radius`` on a given dimension ``dim``.

    .. Note:: this is the functional implementation of :class:`BallSampler`.

    :param dim: dimension of the vectors, denoted as :math:`d`.
    :param samples: sample size, denoted as(dh[:,:, :-1] + dw[:,:-1, :]).pow(1/self.a).sum() :math:`n`.
    :param radius: radius of the ball.
    :param device: device on which the computations should be executed. Default is ``'cpu'``.
    :param generator: the pseudo-random generator :class:`torch.Generator` to specify. If None, a new
        `torch.Generator` is created on the specified device.

    :returns: a :class:`torch.Tensor` with shape :math:`(n, d)` such that their :math:`L^2` norm is
        less or equal to ``radius``.
    """
    if (not isinstance(dim, int)) and (dim < 1):
        raise ValueError("The dimension must be an integer greater than 1")

    if generator is None:
        generator = torch.Generator(device=device)

    Z = torch.randn(samples, dim, device=device, generator=generator)
    sphere = torch.nn.functional.normalize(Z)
    radius = radius * (torch.rand(samples, 1, device=device, generator=generator) ** (1 / dim))
    delta = sphere * radius
    return delta
