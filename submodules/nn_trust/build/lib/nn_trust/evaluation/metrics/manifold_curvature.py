from typing import Optional

import torch
from pydantic import Field

from nn_trust import Task
from nn_trust.evaluation._statistics import Statistic, StatisticConfig
from nn_trust.evaluation.statistic_factory import StatisticsFactory
from ._utils import _compute_curvature


class ManifoldCurvatureConfig(StatisticConfig):
    n_neighbours: int = Field(
        default=15,
        description="Number of neighboring points used to estimate the manifold's curvature. Default is ``15``.",
        ge=1,
        title="Neighbours"
    )
    percentile: float = Field(
        default=0.1,
        description="Percentile of weighted angles of data point for each label to store.",
        ge=0.0,
        le=1.0,
        title="Percentile"
    )
    eps: float = Field(
        default=1e-8,
        description="Regularization parameter for numerical stability purposes",
        title="Regularization parameter"
    )


@StatisticsFactory.register(
    name="Manifold curvature",
    description="It computes the manifold curvature for each label class. The manifold curvature is averaged over all points in the respective manifold. The output value ranges in :math:`[0, \pi]`, the higher value means the higher curvature, hence the less stable the model output might be.",
    actions={"aggregator"},
    task={Task.Classification, Task.Detection, Task.Segmentation}
)
class ManifoldCurvature(Statistic):
    r"""Implementation of *Algorithm 1* applied to the definition of Black-box robustness
    as described in [1]_. It computes the manifold curvature for each label class.
    The manifold curvature is averaged over all points in the respective manifold. The output value
    ranges in :math:`[0, \pi]`, the higher value means the higher curvature, hence the less stable the model
    output might be.

    .. Generally, a more robust estimate is to consider the median instead of the average. For this reason, the
        state ``worst_angles`` store the ``percentile`` of largest computed weighted angles specified. By using the
        ``worst_indices`` we obtain the highly *irregular* points with respect to the manifold. This, should be
        interesting for exploring datasets and the model's internal representations: highly irregular means that it
        should be easier to exploit the non-robustness of the model in the neighborhood of the specified data points.

    .. Note:: the parameter ``n_neighbours`` is highly influential in the result. A higher value means that
        more points are considered for each neighborhood, therefore increasing the local-angle estimation.
        However, if the points are quite sparse, this could lead to worse approximations since a neighbors of
        a point might be quite far, hence the local angle is not well-represented.

    .. Note:: The number of neighbors required is at least ``2``, therefore the number of points required for each
        class must be at least ``3``. A :class:`ValueError` is raised in this occurrence.

    .. [1] Sekmen, A., Bilgin, B. Manifold-based approach for neural network robustness analysis. Commun Eng 3, 118 (2024). https://doi.org/10.1038/s44172-024-00263-8
    """
    CONFIG_T = ManifoldCurvatureConfig

    # The metric is not differentiable because of SVD operations
    is_differentiable: Optional[bool] = False
    # Lower curvature implies a more regular manifold
    higher_is_better: Optional[bool] = False
    # Every time a new point is added, compute everything.
    full_state_update: bool = True

    def __init__(self, config: ManifoldCurvatureConfig):
        super().__init__(config)
        if self.config.n_neighbours < 2:
            raise ValueError("The number of neighbors (n_neighbours) must be larger than 2.")

        self.add_state("pred_points", default=[], dist_reduce_fx="cat")

    @torch.no_grad()
    def update(self, out: torch.Tensor = None, **kwargs) -> None:
        r"""
        :param out: prediction of the network on a given parameters.
        """

        self.pred_points.append(out)

    @torch.no_grad()
    def compute(self) -> float:
        r"""Computes the manifold curvature angles."""
        # tensor_of_points has shape (d, n_points)
        pred_points = torch.cat(self.pred_points)
        if len(pred_points) == 0:
            return 0.0
        else:
            return _compute_curvature(
                pred_points=pred_points,
                n_neighbours=self.config.n_neighbours,
                eps=self.config.eps,
                device=self.config.device
            )
