import torch
from ultralytics import YOLO

from nn_trust.core import ModelAdapter, Knowledge
from nn_trust.attack.detection_utils import relativize_


class UltralyticsModel(ModelAdapter):
    """
    Adapt the ultralytics/ultralytics package by exploiting the raw models of the yolo, getting all the prediction in
    the desired format, and also propagating the gradients and the class scores.
    """

    def __init__(self,
                 model_name: str = 'yolov8n.pt',
                 threat_model: Knowledge = Knowledge.White,
                 ) -> None:
        """
        Initialize the adapter of ultralytics model, wrapping
        """
        model = YOLO(model_name).model
        super().__init__(model, threat_model)

    def forward(self,
                x: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(x.size()) == 4, "Expected 4D tensor (N * C * H * W)."
        preds = super().forward(x)
        # Extract boxes and classification prediction
        boxes = preds[0][:, :].permute(0, 2, 1)
        relativize_(boxes, x.size(3), x.size(2))
        return (boxes[:, :, :4], boxes[:, :, 4:])
