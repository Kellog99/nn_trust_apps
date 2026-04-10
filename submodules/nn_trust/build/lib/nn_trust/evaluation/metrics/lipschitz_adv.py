import math
from collections.abc import Callable
from typing import Optional

import torch
import torch.nn as nn
from pydantic import Field

from nn_trust.evaluation._statistics import Statistic, StatisticConfig
from nn_trust.evaluation.statistic_factory import StatisticsFactory

# Type of function that maps a torch backward operation, e.g. AddmmBackward0, to a function that takes as
# argument a list of tensors and returns a new tensor. The idea being that maps the operations of the computational
# graph to function that can then be reduced.
ReduceFunction = Callable[[list[float | int]], float | int]
MapBackwardOpToReduceFunction = Callable[["torch.BackwardOperations"], ReduceFunction]


def _map_and_reduce_on_backward_grad_graph(
        grad_fn: "torch.BackwardGradGraph",
        mapping_function: MapBackwardOpToReduceFunction
) -> float:
    r"""Takes a backward computation graph as generated from PyTorch and reduce
    it with a mapping operation of specified type.

    :param grad_fn: the backward computational graph.
    :type grad_fn: torch.BackwardGradGraph.
    :param mapping_function: a function that maps each node of the computational graph to a function of type
        `ReduceFunction`.
    :type mapping_function: MapBackwardOpToReduceFunction.

    :raises ValueError: if *grad_fn* is None.

    :returns: A floating point.
    """
    if grad_fn[0] is None:
        raise ValueError("The graph cannot be None.")

    values = []
    for child in grad_fn[0].next_functions:
        if child[0] is None:
            continue
        value = mapping_function(child[0])([_map_and_reduce_on_backward_grad_graph(child, mapping_function)])
        values.append(value)

    return mapping_function(grad_fn[0])(values)


def _from_backwards_to_operation(node, **kwargs) -> ReduceFunction:
    r"""Maps all backward operations to python functions.

    :param kwargs: the field 'norm' specifies which norm to apply to each weight matrix.

    :returns: ReduceFunction.
    """
    if "norm" in kwargs:
        norm_func = lambda x: kwargs["norm"](x).item()
    else:
        norm_func = lambda x: torch.norm(x, p=2.0).item()

    if node is None:
        raise ValueError("The node can't be None.")
    elif "AddBackward" in node.name():
        return sum
    # Consider only the weights used in matrix multiplication and avoid the bias term norm
    elif hasattr(node, "variable") and node.variable.dim() > 1:
        return lambda args: norm_func(node.variable.detach())
    elif "ReluBackward" in node.name() or "TanhBackward" in node.name():
        return math.prod
    elif "SigmoidBackward" in node.name():
        return lambda args: 0.25 * math.prod(args)
    else:
        return math.prod


def compute_upper_bound_lipschitz_constant(network: nn.Module, sample_input: torch.Tensor, **kwargs) -> float:
    r"""
    Uses the backward propagation algorithm to compute an upper
    estimate on the Lipschitz constant of a given model.

    Given a model :math:`f : \mathbb{R}^d \to \mathbb{R}^s`, we compute an upper bound
    for :math:`K` such that for any choice :math:`x, y \in \mathbb{R}^d`,
    we have

    .. math::
        \| f(x) - f(y) \| \le K \|x-y\|

    Note: The argument `sample_input` is not influential in the computation of the Lipschitz constant
    and it is used only to generate the backward graph.

    :param network: A PyTorch module.
    :type network: :class:`nn.Module`
    :param sample_input: An input value compatible with the input of the *network* parameter.
    :type sample_input: :class:`torch.Tensor`
    :param kwargs: A dictionary containing additional settings for the computation of the Lipschitz constant:
            - the field 'norm' specifies which norm to apply to each weight matrix.


    Example::

        Consider a simple feedforward network, then we can compute an upper bound for the Lipschitz constant
        as follows:

        >>> torch.manual_seed(1234)
        >>> example_net = nn.Sequential(*([nn.Linear(3, 3, bias=False)] * 3))
        >>> example_input = torch.rand((1, 3))
        >>> compute_upper_bound_lipschitz_constant(example_net, example_input)
        1.0492145164978255
    """
    # Construct the backward graph
    network.requires_grad_(True)
    network.zero_grad()
    val = network(sample_input).sum()
    val.backward()

    # Maps and reduce the computation on the backward graph
    evaluated_norm = _map_and_reduce_on_backward_grad_graph(
        grad_fn=(val.grad_fn, -1),
        mapping_function=lambda x: _from_backwards_to_operation(x, kwargs=kwargs)
    )
    return evaluated_norm


def compute_adversary_lipschitz_constant(
        network: nn.Module,
        sample: torch.Tensor,
        adversary_sample: torch.Tensor,
        min_clamp_toll: float = 1e-9
):
    r"""Estimates the Lipschitz constant of a ``network`` with respect to the given
    ``sample`` and ``adversary_sample``.

    :param network: a module that takes in input the ``sample`` and ``adversary_sample`` as-is.
    :type network: :class:`torch.nn.Module`
    :param sample: a batch of inputs.
    :type sample: :class:`torch.Tensor`
    :param adversary_sample: a batch of adversary inputs.
    :type adversary_sample: :class:`torch.Tensor`
    :param min_clamp_toll: Values that fall below the threshold are constrained to the minimum ``min_clamp_toll``.
        Default is ``1e-9``.
    :type min_clamp_toll: ``float``

    :returns: a tensor of shape :math:`B` with ``B`` being
    """
    # Computes the output of the network wrt the sample and the adversary samples.
    out_sample = network(sample)
    adv_out_sample = network(adversary_sample)
    # Computes all but the batch dimension
    out_dims = list(range(1, out_sample.dim()))
    sample_dims = list(range(1, sample.dim()))
    # Computes the quotient of the diff norms
    out_norm_diff = torch.norm(out_sample - adv_out_sample, p=2.0, dim=out_dims)
    adversary_norm_diff = torch.norm(sample - adversary_sample, p=2.0, dim=sample_dims).clamp(min=min_clamp_toll)
    return out_norm_diff / adversary_norm_diff


class AdversaryLipschitzBoundConfig(StatisticConfig):
    model: nn.Module = Field(
        default=...,
        description="Model on which Lipschitzianity is to be calculated with adversarial inputs.",
        title="Model"
    )


@StatisticsFactory.register(
    name="Adversary Lipschitz",
    description="Uses adversary samples to estimate an upper bound of the Lipschitz constant of a given model.",
    actions={"performance"}
)
class AdversaryLipschitzBound(Statistic):
    r"""Uses adversary samples to estimate an upper bound of the Lipschitz constant of
    a given model. In general a value smaller than :math:`1` is optimal, as it means that
    the model behaves *sub-linearly*.

    Given a model :math:`f : \mathbb{R}^d \to \mathbb{R}^s`, we compute an estimate
    for :math:`K` such that for any choice :math:`x, y \in \mathbb{R}^d`, we have

    ... math::
        \| f(x) - f(y) \| \le K \|x-y\|

    ... Note:: This metric is versatile: it is applicable to both black-box and white-box scenarios.
        Its effectiveness is independent of the threat model, as the adversary samples are tied to the
        generation method used. For example, for a white-box threat model an algorithm as PGD


    :param model: a model.
    :type model: :class:`nn_trust.attack.AttackedModel`
    """

    # The metric is not differentiable
    is_differentiable: Optional[bool] = False
    # Lower lipschitz constant implies a more stable model
    higher_is_better: Optional[bool] = False
    # Every time a new point is added, compute everything.
    full_state_update: bool = True

    CONFIG_T = AdversaryLipschitzBoundConfig

    def __init__(self, config: AdversaryLipschitzBoundConfig):
        super().__init__(config)
        self.add_state("lipschitz_bounds", default=[], dist_reduce_fx="sum")

    def update(self, data: torch.Tensor, adv_data: torch.Tensor) -> None:
        res = compute_adversary_lipschitz_constant(self.config.model, data, adv_data)
        self.lipschitz_bounds.extend(res.unbind(0))

    def compute(self):
        return max(self.lipschitz_bounds)
