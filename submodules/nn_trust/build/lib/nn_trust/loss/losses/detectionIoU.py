import torch
import torchvision
from pydantic import Field

from nn_trust import Task
from nn_trust.loss._loss import Loss, LossConfig
from nn_trust.loss.loss_factory import LossFactory


class DetectionMisclassificationLossConfig(LossConfig):
    threshold: float = Field(
        default=1e-2,
        description="Threshold to set for the bbox.",
        gt=0.0,
        title="Threshold"
    )


@LossFactory.register(
    name="Detection Misclassification Loss",
    description="Bounding box loss.",
    task={Task.Detection}
)
class DetectionIoUBoundingBoxLoss(Loss):
    CONFIG_T = DetectionMisclassificationLossConfig

    def forward(self,
                out_adv: torch.Tensor,
                out_adv_bboxes: torch.Tensor,
                target_bboxes: torch.Tensor,
                **kwargs) -> torch.Tensor:
        # In this case the tensor of the bboxes have shape respectively (B, D, 4) and (B, D', 4)

        # select best adv boxes
        best_boxes = out_adv.max(dim=-1).values > self.config.threshold
        ious = torchvision.ops.iou(out_adv_bboxes[best_boxes], target_bboxes)
        return self.reduce(ious)
