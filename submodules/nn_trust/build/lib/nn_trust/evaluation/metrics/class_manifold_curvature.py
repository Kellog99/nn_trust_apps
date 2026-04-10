from typing import Optional

import torch
from pydantic import Field

from nn_trust import Task
from nn_trust.evaluation._statistics import Statistic, StatisticConfig
from nn_trust.evaluation.metrics._utils import _compute_curvature
from nn_trust.evaluation.statistic_factory import StatisticsFactory


class ClassManifoldCurvatureConfig(StatisticConfig):
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
        gt=0,
        title="Regularization"
    )


@StatisticsFactory.register(
    name="Class Manifold curvature",
    description="It computes the manifold curvature for each class. The manifold curvature is averaged over all points in the respective manifold. The output value ranges in :math:`[0, \pi]`, the higher value means the higher curvature, hence the less stable the model output might be.",
    actions={"aggregator"},
    task={Task.Classification, Task.Detection, Task.Segmentation}
)
class ClassManifoldCurvature(Statistic):
    r"""
    Is the same algorithm as the one above but this performs the computation for each class
    """
    CONFIG_T = ClassManifoldCurvatureConfig
    # The metric is not differentiable because of SVD operations
    is_differentiable: Optional[bool] = False
    # Lower curvature implies a more regular manifold
    higher_is_better: Optional[bool] = False
    # Every time a new point is added, compute everything.
    full_state_update: bool = True

    def __init__(self, config: ClassManifoldCurvatureConfig):
        super().__init__(config)
        if self.config.n_neighbours < 2:
            raise ValueError("The number of neighbors (n_neighbours) must be larger than 2.")

        self.add_state("label", default=[], dist_reduce_fx="cat")
        self.add_state("pred_points", default=[], dist_reduce_fx="cat")

    @torch.no_grad()
    def update(self, out: torch.Tensor = None, **kwargs) -> None:
        r"""Updates the :class:`ManifoldCurvature` list of points given a batch of predictions and y.

        :param out: batch output of a model for a ``target`` class.
        """
        if out is not None:
            self.label.append(out.argmax(-1))
            self.pred_points.append(out)

    @torch.no_grad()
    def compute(self) -> list[float]:
        r"""Computes the manifold curvature angles for each class."""
        label = torch.cat(self.label)
        pred_points = torch.cat(self.pred_points)

        out = []
        for i in range(self.config.num_classes):
            pred_points_i = pred_points[label == i]
            out.append(
                _compute_curvature(
                    pred_points=pred_points_i,
                    n_neighbours=self.config.n_neighbours,
                    eps=self.config.eps,
                    device=self.config.device
                )
            )
        return out
