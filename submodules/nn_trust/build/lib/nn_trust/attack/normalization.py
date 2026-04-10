import abc
from math import isnan
from typing import List, NoReturn, Optional

import torch
import torch.fft as fft


class Normalization(abc.ABC):
    @abc.abstractmethod
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given a tensor ``x``, returns a normalization based on the normalization method.

        :param x: a :class:`torch.Tensor`.

        :returns: a :class:`torch.Tensor` with the same shape of ``x``.
        """

    @abc.abstractmethod
    def normalize_(self, x: torch.Tensor) -> None:
        """
        In-place version of the `normalize` function.

        :param x: a :class:`torch.Tensor`.
        """

    def __call__(self, *args, **kwargs):
        return self.normalize(*args)

    def __repr__(self) -> str:
        return f"{type(self).__name__}: " + str({k: self.__dict__[k] for k in self.__dict__})


class LpNormalization(Normalization):
    """Defines a normalization method by scaling a tensor by a constant value,
    such that the result should be a tensor belonging to a sphere in :math:`L^p` of
    specified radius and centered in :math:`0`.

    :param p: float value corresponding to the :math:`L^p` norm to use.
        Note that the infinity-norm corresponds to the max-norm.
    :param radius: float value corresponding to the radius of the sphere in
        the appropriate :math:`L^p` space.
    :param center: a tensor of the same shape (or broadcasted shape) of the
        tensor in input of the implemented method representing the center of the
        :math:`L^p` sphere we want to project to.

    Examples::

    Normalization with respect to an L^2 unitary sphere centered in 0 of a given input x.

    >>> x = torch.rand((10, 20))
    >>> norm = LpNormalization(p=2.0, radius=1.0)
    >>> norm.normalize(x)

    Normalization with respect to an L^2 unitary sphere centered in C of a given input x.

    >>> C = torch.eye(10)
    >>> x = torch.rand((10, 10))
    >>> norm = LpNormalization(p=2.0, radius=1.0)
    >>> norm.normalize(x - C)

    Similarly to the previous example, one can pass the center of the normalization as
    a parameter. Then,

    >>> C = torch.eye(10)
    >>> x = torch.rand((10, 10))
    >>> norm = LpNormalization(p=2.0, radius=1.0, center=C)
    >>> norm.normalize(x)

    is equivalent to the previous block of code.
    """

    def __init__(self, p: float, radius: float, center: Optional[torch.Tensor] = None, batched: bool = False):
        if p < 1.0 or isnan(p):
            raise ValueError(f"The value of 'p' ({p}) must be a float larger or equal to 1.")
        if radius < 0.0 or isnan(radius):
            raise ValueError(f"The value of 'radius' ({radius}) must be a non-negative float value.")

        self.p = p
        self.radius = radius
        self.center = center
        self.batched = batched

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Projection onto L^inf norm
        v = x.clone()
        self.normalize_(v)
        return v

    def normalize_(self, x: torch.Tensor) -> None:
        if self.center is not None:
            x.data -= self.center

        # Projection onto L^inf norm
        if self.p == float("inf"):
            above_threshold = x.abs() >= self.radius
            x[above_threshold] = torch.sign(x[above_threshold]) * self.radius
        else:
            if self.batched:
                norms = x.norm(p=self.p, dim=tuple(range(1, x.dim())), keepdim=True)
            else:
                norms = x.norm(p=self.p)
            x.div_(norms.clamp_min(1e-12))
            x.mul_(self.radius)


class ClampNormalization(Normalization):
    """Defines a normalization method by clamping every value of a tensor in
    a range ``[min_val, max_val]``.

    Note: this method should be equivalent, when ``-min_val = max_val``,
    to :class:`LpNormalization` with ``p='inf'`` and ``radius = max_val``.

    :param min_val: float value to clamp the value below this threshold.
    :param max_val: float value to clamp the value above this threshold.

    Examples::

    Clamping all values of a given input ``x`` between :math:`[-1, 3]`.

    >>> x = torch.rand((10, 20))
    >>> norm = ClampNormalization(min_val=-1, max_val=3)
    >>> norm.normalize(x)

    Equivalently using LpNormalization with ``p=float('inf')``

    >>> x = torch.rand((10, 20))
    >>> norm = LpNormalization(p=float("inf"), radius=2.0)
    >>> norm.normalize(x - 1.0)
    """

    def __init__(self, min_val: float, max_val: float):
        if isnan(min_val) or isnan(max_val):
            raise ValueError(
                f"Both 'min_val' ({min_val}) and 'max_val' ({max_val}) must be valid floating point values."
            )

        if min_val >= max_val:
            raise ValueError(f"The range must be of positive length, but the set range is [{min_val}, {max_val}].")

        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.min_val, self.max_val)

    def normalize_(self, x: torch.Tensor) -> NoReturn:
        torch.clamp_(x, self.min_val, self.max_val)


class SignNormalization(Normalization):
    """Defines a normalization method by computing the sign of each element of the
    given tensor. The value for the element :math:`0` can be specified via the parameter
    ``zero_value``.

    :param zero_value: value of the sign of the zero element. Default is ``1.0``.

    Example::

    Given an input tensor, returns the sign for each component.

    >>> x = torch.arange(10) - 5
    >>> norm = SignNormalization(zero_value=1.0)
    >>> norm.normalize_(x)
    >>> assert x.sum() == 0.0
    """

    def __init__(self, zero_value: float = 1.0):
        super()
        self.zero_value = zero_value

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        res = torch.sign(x).data
        return torch.where(res == 0, self.zero_value, res)

    def normalize_(self, x: torch.Tensor) -> None:
        x.sign_()
        x.data = x.where(x != 0, self.zero_value).data


class FrequencyThresholdNormalization(Normalization):
    r"""Thresholds the frequencies values of the discrete fourier transform of a given input.
    The idea being that we threshold to a given value all the frequencies corresponding lower
    or higher than a certain value. More specifically, assume ``x`` is a :math:`1` dimensional vector of length
    :math:`N` (which we assume as a signal), we denote with :math:`X_k` the coefficient for the :math:`k`-th frequency,
    i.e. the Discrete Fourier Transform of the signal :math:`x` for wave with period :math:`(2 pi k) / N`. If :math:`k`
    is comprised between ``low_bandwidth`` and ``high_bandwidth``, its value is preserved, otherwise it is set to
    ``value``.

    :param low_bandwidth: integer value representing the minimum frequency to consider.
    :param high_bandwidth: integer value representing the maximum frequency to consider.
    :param value: value to set the discrete fourier transform outside the region of interest in the frequency domain
        specified by axis, low_bandwidth and high_bandwidth.
    :param axis: defaults to the -1 axis. Axis of the dft of an input tensor on which
        we apply the filtering. For example if ``x`` has shape :math:`(B, W, H)`, ``low_bandwidth=0``,
        ``high_bandwidth=W/2``, ``axis=[1,2]``, then the :math:`n`-dim DFT of ``x`` has shape :math:`(B, W, H)`
        and the values of ``x``'s DFT that resides in the indices :math:`\{ (b,w,h) | \text{low\_bandwidth} \le w \le \text{high\_bandwidth} \land \text{low\_bandwidth} \le h \le \text{high\_bandwidth}\}`
        are preserved, while the others are set to ``value``.

    Example::

    Suppose that we want to filter the frequencies in range :math:`[0, 20]` on the width and height axis
    for an image batch with shape :math:`(B, C, W, H)`. Then, the width and height frequencies are along the axis
    :math:`2,3` , hence we define the :class:`FrequencyThresholdNormalization` class as:

    >>> ftn = FrequencyThresholdNormalization(low_bandwidth=20, high_bandwidth=-1, axis=[2, 3], value=0.0)
    >>> result = ftn(image_batch)

    """

    def __init__(
        self,
        low_bandwidth: int,
        high_bandwidth: int,
        value: Optional[float] = None,
        axis: Optional[int | List[int]] = None,
    ):
        # Note that we allow for the '0' clamping, i.e. when low=high,
        # hence no value is preserved.
        if 0 < high_bandwidth < low_bandwidth:
            raise ValueError(f"{high_bandwidth=} must be higher than {low_bandwidth=}.")
        self.low_bandwidth_filter = low_bandwidth
        self.high_bandwidth_filter = high_bandwidth

        if value is None:
            self.value = 0.0
        else:
            self.value = value

        # convert the axis parameter to a list of indexes
        if axis is None:
            self.axis = [-1]
        elif type(axis) is list:
            self.axis = axis
        elif type(axis) is int:
            self.axis = [axis]
        else:
            raise ValueError(f"The chosen axis is not compatible with this implementation: {axis=}.")

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # first compute the dft of x
        fftx = fft.fftn(x)
        # compute the masking for the filtering
        mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)
        for ax in self.axis:
            ax_size = x.size(ax)
            # takes the positive examples on the given axis
            ax_indices = torch.arange(ax_size, dtype=torch.int, device=x.device)
            # account for negative index
            upper_filter = self.high_bandwidth_filter
            if self.high_bandwidth_filter < 0:
                upper_filter += ax_size
            pos_mask = torch.logical_and(ax_indices >= self.low_bandwidth_filter, ax_indices <= upper_filter)
            # takes the positive example on the specified axis, by storing it in the mask
            mask = mask.moveaxis(ax, destination=0)
            mask[pos_mask] = True
            mask = mask.moveaxis(0, destination=ax)
        # actually clamp by setting to 0, the discrete fourier transform of x
        clamped_fftx = torch.where(mask, fftx, self.value)
        # return the real part of the inverse dft of the clamped dft of x
        return fft.ifftn(clamped_fftx).real

    def normalize_(self, x: torch.Tensor) -> None:
        x.data = self.normalize(x).data
