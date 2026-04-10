import torch

from nn_trust import Task
from nn_trust.loss._loss import Loss, LossConfig
from nn_trust.loss.loss_factory import LossFactory


class DetectionMisclassificationLossConfig(LossConfig):
    pass


@LossFactory.register(
    name="Detection Misclassification",
    description="This loss aims to measure the discrepancy between the model's expected codomain and its output.",
    task={Task.Classification, Task.Detection}
)
class DetectionMisclassificationLoss(Loss):

    def forward(self, out_adv: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        # in this case the tensor has shape (B, D, C)
        best_indices = out_adv.argmax(dim=-1)[:, :target.shape[1]]
        targeted = torch.amin(target, dim=tuple(range(1, target.dim())))
        negative_rows = targeted < 0
        loss = torch.zeros_like(target).float()
        # compute something like cosine sim on classification results
        if negative_rows.any():
            loss[negative_rows] = - (out_adv[negative_rows, best_indices] * target[negative_rows])
        if (~negative_rows).any():
            loss[~negative_rows] = (out_adv[~negative_rows, best_indices] * target[~negative_rows])
        return self.reduce(loss)
