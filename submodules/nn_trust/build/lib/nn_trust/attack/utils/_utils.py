import warnings
from collections.abc import Callable
from typing import Any, Literal

import torch
import torch.nn as nn
from pydantic import BaseModel


def get_min(element1: list[Any] | float, element2: list[Any] | float) -> list[Any] | float:
    if isinstance(element1, float):
        if isinstance(element2, float):
            return min(element2, element1)
        elif isinstance(element2, list):
            return [min(elm, element1) for elm in element2]
        else:
            raise ValueError("The type of the second element is not supported.")
    elif isinstance(element1, list):
        return [get_min(el, element2) for el in element2]
    else:
        raise ValueError("The type of the first element is not supported.")


def enumerated_list(obj: list | dict | BaseModel, enumeration: bool = True) -> str:
    """
    A nice representation for printing all the elements inside a list
    """
    if isinstance(obj, list):
        if enumeration:
            lst = [f"{i + 1}.) {item} \n" for i, item in enumerate(obj)]
        else:
            lst = [f"* {item} \n" for item in obj]
        out = "".join(lst)
        out = "\n" + out
        return out

    elif isinstance(obj, dict):
        return enumerated_list([(key, value) for key, value in obj.items()], enumeration=enumeration)
    elif isinstance(obj, BaseModel):
        return enumerated_list(obj.__dict__)
    else:
        raise ValueError("The type of obj is not supported.")


def to_device(
        obj: list | dict | torch.Tensor | nn.Module | BaseModel,
        device: torch.device,
) -> list | dict | torch.Tensor:
    """
    Move a composite object to a device in a recursive way. Works with list, dict and torch.Tensor.

    :param obj: The object to move.
    :param device: The device to move the object to.

    :return: The moved object.
    """
    if isinstance(obj, torch.Tensor | nn.Module):
        obj = obj.to(device)
    elif type(obj) is list:
        for i in range(len(obj)):
            obj[i] = to_device(obj[i], device)
    elif type(obj) is dict:
        for key in obj:
            obj[key] = to_device(obj[key], device)
    elif isinstance(obj, BaseModel):
        for field_name, field_value in obj:
            setattr(obj, field_name, to_device(field_value, device))
    return obj


def _compare_misclassification(
        y_label: torch.Tensor,
        y_pred: torch.Tensor,
        dim: int = -1,
        reduction: Literal["any", "all", "none"] = "none"
) -> torch.Tensor:
    """Return a bool tensor where the attack succeeded.

    :param y_label: The target labels tensor (batch)
    :param y_pred: The predicted labels tensor (batch)
    :param dim: The dimension to compare the labels
    :param reduction: The reduction on the list of booleans
    :return: A bool tensor where is true for the attacks that have worked
    """
    # Compute which element should be evaded because the label is negative
    row_mins = torch.amin(y_label, dim=tuple(range(1, y_label.dim())))
    is_targeted = row_mins < 0
    # mask = is_targeted.view(-1, *[1] * (y_label.dim() - 1)).expand_as(y_label)

    # Compute the correct classification
    results = torch.empty(y_label.shape[:-1], dtype=torch.bool, device=y_label.device)
    if is_targeted.any():
        results[is_targeted] = y_label[is_targeted].abs().argmax(dim=dim) != y_pred[is_targeted].argmax(dim=dim)
    if (~is_targeted).any():
        results[~is_targeted] = y_label[~is_targeted].abs().argmax(dim=dim) == y_pred[~is_targeted].argmax(dim=dim)

    if reduction == "all":
        results = torch.all(results)
    elif reduction == "any":
        results = torch.any(results)
    return results


def _compare_semantic(
        y_label: torch.Tensor, y_pred: torch.Tensor, dim: int = -1, reduction: Literal["any", "all", "none"] = "none"
) -> torch.Tensor:
    """Compare label with prediction in the semantic segmentation task.

    :param y_label: A 1 * H * W label tensor.
    :param y_pred: A 1 * H * W output tensor.
    :param dim: The dimension to compare the labels
    :param reduction: The reduction on the list of booleans

    :return: A bool tensor where each pixel will be true if the attacks have worked there.
    """
    # Compute which element should be evaded because the label is negative
    row_mins = torch.amin(y_label, dim=tuple(range(1, y_label.dim())))
    is_targeted = row_mins < 0
    mask = is_targeted.view(-1, *[1] * (y_label.dim() - 1)).expand_as(y_label)

    # Compute the correct classification
    results = torch.empty(y_label.shape[:-1], dtype=torch.bool, device=y_label.device)
    if mask.any():
        results[is_targeted] = torch.logial_or(
            y_label[is_targeted].abs().argmax(dim=dim) != y_pred[is_targeted].argmax(dim=dim),
            y_label[is_targeted].abs().argmax(dim=dim) == 0,
        )
    if (~mask).any():
        results[~is_targeted] = y_label[~is_targeted].abs().argmax(dim=dim) == y_pred[~is_targeted].argmax(dim=dim)

    if reduction == "all":
        results = torch.all(results)
    elif reduction == "any":
        results = torch.any(results)
    return results


def _binary_line_search(
        x_min: float,
        x_max: float,
        objective_func: Callable[[float], bool],
        max_iters: int = 100,
        epsilon: float = 1.0e-6,
        verbose: bool = False,
) -> float:
    """
    Minimizes the parameter ``x`` in a range ``[x_{min}, x_{max}]`` for which the
    objective function is satisfied. The goal of this function is to compute the
    minimal value of x in ``[x_min, x_max]`` such that ``f(x)`` is ``True``, where ``f`` is the
    ``objective_func``. The procedure is fairly simple and consists of iteratively
    halve the search space.

    :param x_min: minimum value of the parameter to optimize
    :param x_max: maximum value of the parameter to optimize
    :param objective_func: functional whose validation is required to meet the optimal condition.
    :param max_iters: integer counting the maximum admissible iterations of the optimization procedure.
    :param epsilon: float representing the maximum error on the solution.

    :returns: the solution of the minimization procedure.
    """
    if objective_func(x_min):
        return x_min

    if max_iters == 0:
        return x_min + (x_max - x_min) / 2.0

    for iters in range(max_iters):
        x_mid = x_min + (x_max - x_min) / 2.0

        if objective_func(x_mid):
            x_max = x_mid
        else:
            x_min = x_mid

        if abs(x_max - x_min) < 2.0 * epsilon:
            if verbose:
                print(f"Iter: {iters}; Error: {abs(x_max - x_min)}")
            return x_mid

    if verbose:
        print(f"Iter: {iters}; Error: {abs(x_max - x_min)}")

    return x_mid


def _binary_line_search_geometric(
        initial_value: float,
        alpha: float,
        objective_func: Callable[[float], bool],
        max_iters: int = 100,
        epsilon: float = 1e-6,
        verbose: bool = False,
) -> float:
    """
    Same as above, just use a dynamical reshaping interval by using a parameter alpha.

    See Algorithm 1 of [1] for the description of the optimization procedure.

    [1] https://doi.org/10.48550/arXiv.1807.04457
    """
    queries = 0
    if not objective_func(initial_value):
        queries += 1
        v_left, v_right = initial_value, (1 + alpha) * initial_value
        while queries < max_iters:
            if objective_func(v_right):
                break
            v_right *= 1 + alpha
            queries += 1
    else:
        queries += 1
        v_left, v_right = (1 - alpha) * initial_value, initial_value
        while queries < max_iters:
            if not objective_func(v_left):
                break
            v_left *= 1 - alpha
            queries += 1

    if verbose:
        print(f"[{v_left=}, {v_right=}]")

    return _binary_line_search(v_left, v_right, objective_func, max_iters=max_iters - queries, epsilon=epsilon)


@torch.no_grad()
def batched_argmin_line_search(
        model: nn.Module,
        evaluator: Callable[[torch.Tensor], torch.Tensor],
        x0: torch.Tensor,
        dx: torch.Tensor,
        min_t: float | torch.Tensor,
        max_t: float | torch.Tensor,
        max_iters: int = 200,
        max_eps: float = 1e-6,
) -> torch.Tensor:
    r"""Implement an argmin line search approach on a batch of values given an initial point and direction and
    a given initial scaling of such direction.
    It returns
    .. math::
        t = \textrm{argmin}_{t \in [t_min, t_max]} \text{evaluator}(\text{model}(x_0 + t \cdot d_x)) = True

    via a sequence of subsequent halving, we find the optimal value of t. Hence, in :math:`\log_2(1/\varepsilon)`
    steps we should achieve an :math:`(t_{\text{max}} - t_\text{min})\varepsilon` error for each :math:`t`.

    .. Note: this routine is optimized for batched input with ``x0`` of shape :math:`(B, V_1, \dots, V_k)`
        and ``dx`` :math:`(B, V_1, \dots, V_k)`.

    :param model: a :class:`torch.nn.Module` that maps the data :math:`x_0 + td_x` into something meaningful
        for the ``evaluator`` function. It expects a batched input.
    :param evaluator: must return a :class:`torch.boolTensor`. It expects a batched input compatible with the
        ``model``'s output.
    :param x0: initial point chosen for the optimization procedure.
    :param dx: optimization direction.
    :param min_t: either the minimum value of ``t`` available or a tensor of the minimum values of ``t`` for each
        element of the batch ``x0``.
    :param max_t: either the maximum value of ``t`` available or a tensor of the maximum values of ``t`` for each
        element of the batch ``x0``.
    :param max_iters: maximum number of iterations.
    :param max_eps: maximum error before stopping the optimization procedure.

    :returns: a tensor that corresponds to the smallest value of ``t`` along the optimization direction
        ``dx`` that preserves the evaluation of the ``evaluator`` to ``True``.
    """
    # Tries to find the argmin
    if x0.shape != dx.shape:
        raise ValueError("The shape of the input and the shape of the optimization direction are not compatible")

    # Extend the float to a tensor of values.
    if isinstance(max_t, float) or isinstance(max_t, int):
        max_t = torch.tensor([max_t] * x0.shape[0], dtype=x0.dtype)

    if isinstance(min_t, float) or isinstance(min_t, int):
        min_t = torch.tensor([min_t] * x0.shape[0], dtype=x0.dtype)

    broadcast_shape = (-1, *([1] * (dx.dim() - 1)))
    # Halving stuff and pass to a tensor format to the correct device
    max_ts = max_t.to(x0.device)
    min_ts = min_t.to(x0.device)
    ts = max_ts / 2.0 + min_ts / 2.0
    either_bounds = torch.logical_or(
        evaluator(model(x0 + dx * max_ts.view(*broadcast_shape))),
        evaluator(model(x0 + dx * min_ts.view(*broadcast_shape))),
    )
    if not torch.all(either_bounds):
        warnings.warn(
            "We can't guarantee convergence to a meaningful output because the evaluator is not correct on either of its boundary points.",
            UserWarning,
        )

    # Mask the already achieved minima
    mask_dir_batch = torch.ones(x0.shape[0], dtype=bool, device=x0.device)
    for i in range(max_iters):
        # If we don't have to optimize anymore, stop.
        if all(torch.logical_not(mask_dir_batch)):
            break

        translated_x = x0[mask_dir_batch] + ts[mask_dir_batch].view(*broadcast_shape) * dx[mask_dir_batch]
        # In case we lose a dimension because we have only a single element in the batch
        if translated_x.dim() < x0.dim():
            translated_x = translated_x.unsqueeze(0)

        mask = evaluator(model(translated_x))
        # If the ``translated_x`` already satisfies all the given constraints, stop.
        if all(mask):
            break

        # Copy accordingly to the first mask
        maxts = max_ts[mask_dir_batch]
        midts = ts[mask_dir_batch]
        mints = min_ts[mask_dir_batch]

        # If the evaluator is True, then that coordinate can be optimized further
        # by considering a lower value of ts
        maxts[mask] = midts[mask]
        max_ts[mask_dir_batch] = maxts
        midts[mask] = mints[mask] / 2.0 + midts[mask] / 2.0

        # If the evaluator is False, then that coordinate can be optimized further
        # by considering a higher value of ts
        mask.data = torch.logical_not(mask)
        mints[mask] = midts[mask]
        min_ts[mask_dir_batch] = mints
        midts[mask] = midts[mask] / 2.0 + maxts[mask] / 2.0
        # eventually update the mid-points yet to optimize
        ts[mask_dir_batch] = midts

        mask_dir_batch = (max_ts - min_ts).abs() >= max_eps

    return ts


@torch.no_grad()
def boundary_search(
        model: nn.Module,
        evaluator: Callable[[torch.Tensor], torch.Tensor],
        x0: torch.Tensor,
        dx: torch.Tensor,
        step_size: float = 0.75,
        max_iters: int = 200,
) -> torch.Tensor:
    r"""Implement a line search approach on a batch of values given an initial point and direction and
    tries to compute the minimal value ``t`` such that the ``evaluator`` of the ``model`` output is
    verified to be ``True``.

    It returns
    .. math::
        t^\ast =  \text{evaluator}(\text{model}(x_0 + t^\ast \cdot d_x)) = True

    .. Note: this routine is optimized for batched input with ``x0`` of shape :math:`(B, V_1, \dots, V_k)`
        and ``dx`` :math:`(B, V_1, \dots, V_k)`.

    :param model: a :class:`torch.nn.Module` that maps the data :math:`x_0 + td_x` into something meaningful
        for the ``evaluator`` function. It expects a batched input.
    :param evaluator: must return a :class:`torch.boolTensor`. It expects a batched input compatible with the
        ``model``'s output.
    :param x0: initial point chosen for the optimization procedure.
    :param dx: optimization direction.
    :param step: step size.
    :param max_iters: maximum number of iterations.

    :returns: a tensor value that corresponds to the value ``t`` along the optimization direction
        ``dx`` that preserves the evaluation of the ``evaluator`` of the model's output to ``True``.
    """
    # Tries to find the argmin
    if x0.shape != dx.shape:
        raise ValueError("The shape of the input and the shape of the optimization direction are not compatible")

    ts = torch.ones(x0.shape[0], device=x0.device) * step_size
    broadcast_shape = (-1, *([1] * (dx.dim() - 1)))
    # Mask the already achieved minima
    mask_dir_batch = torch.ones(x0.shape[0], dtype=bool, device=x0.device)
    for _ in range(max_iters):
        translated_x = x0[mask_dir_batch] + ts[mask_dir_batch].view(*broadcast_shape) * dx[mask_dir_batch]
        # In case we lose a dimension because we have only a single element in the batch
        if translated_x.dim() < x0.dim():
            translated_x = translated_x.unsqueeze(0)

        model_eval = model(translated_x)
        mask = evaluator(model_eval)
        # Copy accordingly to the first mask
        midts = ts[mask_dir_batch]
        midts[mask] = midts[mask] + step_size
        # eventually update the ts yet to optimize
        ts[mask_dir_batch] = midts
        # Update the mask
        mask_dir_batch[mask_dir_batch.clone()] = mask

        if all(torch.logical_not(mask_dir_batch)):
            break

    return ts


@torch.no_grad()
def dct_basis_2d(size: int = 224, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    r"""Compute the 2D Discrete Cosine Transform basis.

    :size: size of the basis vectors.
    :device: device onto the computations are executed.

    :returns: a :class:`torch.Tensor` of a matrix whose columns are orthonormal.
    """
    if size <= 0:
        raise ValueError("The size of the basis must be larger than 0.")

    k = -torch.arange(size, device=device) * torch.pi / (2 * size)
    Wr = torch.cos(k)
    Wi = torch.sin(k)
    coeffs = torch.view_as_real(torch.fft.fft(torch.eye(size, device=device), dim=1))
    V = coeffs[..., 0] * Wr + coeffs[..., 1] * Wi
    # normalize
    V[:, 0] /= size ** 0.5
    V[:, 1:] /= (size / 2) ** 0.5
    V = V.view(size, size)
    return V


def stablemax(x, dim=None):
    r"""An implementation of the StableMax function.
    It computes a modified version of the :func:`torch.nn.functional.softmax`
    function that prevents numerical collapse of the :math:`\textrm{SoftMax}`
    function. It evaluates:
        .. math::
            \textrm{StableMax}(x_i) \coloneqq \frac{s(x_i)}{\sum_{j} s(x_j)}

    with
        .. math::
            s(x) \coloneqq \begin{cases}
                x+1 & \text{if} \; x \ge 0\\
                \frac{1}{x-1} & \text{if} \; x < 0
            \end{cases}

    For more information see [1]_.

    :param x: input tensor.
    :type x: :class:`torch.Tensor`.
    :param dim: dimension along which softmax will be computed. Defaults to ``-1``.
    :type dim: ``int``.

    :returns:
    :type: :class:`torch.Tensor`.

    .. [1] Prieto, Lucas, Melih Barsbey, Pedro A. M. Mediano and Tolga Birdal.
    “Grokking at the Edge of Numerical Stability.” (2025).
    """
    if dim is None:
        dim = -1

    mm = x >= 0
    x = x.abs() + 1
    x[~mm] = 1 / x[~mm]
    return x.div(x.sum(dim=dim, keepdims=True))


def stablemax_(x: torch.Tensor, dim: int = -1):
    r"""In-place version of :func:`stablemax`.

    :param x: input tensor.
    :type x: :class:`torch.Tensor`.
    :param dim: dimension along which softmax will be computed. Defaults to ``-1``.
    :type dim: ``int``.

    :returns:
    :type: :class:`torch.Tensor`.
    """
    mm = x >= 0
    x.abs_().add_(1.0)
    x[~mm] = 1 / x[~mm]
    x.div_(x.sum(dim=dim, keepdims=True))
