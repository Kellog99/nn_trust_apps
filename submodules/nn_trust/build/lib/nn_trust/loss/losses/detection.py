from typing import Literal, Optional

import torch
from pydantic import Field

from nn_trust import Task
from nn_trust.loss._loss import Loss, LossConfig
from nn_trust.loss.loss_factory import LossFactory


class DetectionLossConfig(LossConfig):
    combine: Literal["prod", "sum"] = Field(
        default="prod",
        description="It tells how to combine the loss' output.",
        title="Output Combination"
    )
    weight_objectiveness: float = Field(
        default=1.0,
        description="Weight of the objective.",
        title="Weight Objective"
    )
    weight_classes: float = Field(
        default=1.0,
        description="Weight of the classes.",
        title="Classes weight"
    )


@LossFactory.register(
    name="Detection Loss",
    description="The detection loss aims to either fool the object detector in detecting the box or predicting the right class of that box.",
    task={Task.Detection}
)
class DetectionLoss(Loss):
    r"""
    The detection loss aims to either fool the object detector in detecting the box or predicting the right class
    of that box.
    """
    CONFIG_T = DetectionLossConfig

    def forward(
            self,
            obj_scores: torch.Tensor,
            class_scores: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if obj_scores.size(0) == 0 or class_scores.size(0) == 0:
            return torch.tensor(0.0)

        if labels is None:
            class_max_scores = torch.max(class_scores, dim=1)
        else:
            class_max_scores = class_scores[torch.arange(class_scores.size(0)), labels]

        obj_scores *= self.config.weight_objectiveness
        class_max_scores *= self.config.weight_classes

        if self._combine == "prod":
            inner_scores = obj_scores * class_max_scores
        elif self._combine == "sum":
            inner_scores = obj_scores + class_max_scores
        else:
            raise ValueError(f"{self._combine} combine operation not available.")

        return self.reduce(inner_scores)
