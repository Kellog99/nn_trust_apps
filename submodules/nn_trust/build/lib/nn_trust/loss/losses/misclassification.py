from typing import Literal, Callable

import torch
import torch.nn as nn
from pydantic import Field

from nn_trust import Task
from nn_trust.loss._loss import Loss, LossConfig
from nn_trust.loss.loss_factory import LossFactory


class MisclassificationLossConfig(LossConfig):
    toll: float = Field(
        default=0.5,
        description="Tolerance to accept for the new image.",
        gt=0.0,
        title="Tolerance"
    )
    losstype: Literal[0, 1, 2] = Field(
        default=0,
        description="Type of Misclassification loss to use, i.e. 0 = Cross Entropy, 1 = Infomation and 2 =",
        title="Loss type"
    )

    crossentropy: Callable[[...], torch.Tensor] = Field(
        default_factory=lambda: nn.CrossEntropyLoss(reduction="none")
    )


@LossFactory.register(
    name="Misclassification",
    description="The Misclassification loss help the model to misclassify the model's output..",
    task={Task.Classification}
)
class MisclassificationLoss(Loss):
    r"""
    The Misclassification loss help the model to misclassify the model's output.
    There are 3 types of loss:
        1. Entropy loss of those labels to avoid. Let
             .. math::
                p_i = \begin{cases}
                        p_i  & \text{if} \; \text{target}(i) = 1\\
                        0       & \text{if} \; \text{target}(i) = 0,
                    \end{cases}
             Then, the loss is
             .. math::
                -\sum_{j\in J} (1-p_j)\log(1-p_j)

        2. Cross Entropy loss of those labels to avoid
                .. math::
                    p_i = \begin{cases}
                            p_i  & \text{if} \; \text{target}(i) = 1\\
                            0       & \text{if} \; \text{target}(i) = 0,
                        \end{cases}
                Then, the loss is
                .. math::
                    -\sum_{j\in J} \log(1-p_j)

        3. Strong condition of "One vs All". To classify the label :math:`i \in C \setminus J`,
            then it must hold that
            .. math::
                f_i > f_j \forall j \in J

            which is equivalent to
            .. math::
                f_i > max_{j \in J} f_j
            Therefore a stronger condition is the following
            .. math::
                        \min_{i\in C\J} f_i > max_{j\in J} f_j
            Let
            .. math::
            \begin{align*}
                &i^\ast \coloneqq \argmin_{i\in C\J} f_i \\
                &j^\ast \coloneqq \argmin_{j\in J} f_j,
            \end{align*}
            then
            .. math::
                \textrm{loss} = -\log(\sigma(f_{i^\ast}-f_{j^ast}))
    """
    CONFIG_T = MisclassificationLossConfig

    def forward(self, out_adv: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the Misclassification loss of a batch of images

        :param out_adv: The model's output respect ot the adversarial image. It has dimensionality B x C
        :param target:  For the target is a 1D tensor of len B.
                        For the untargeted is a 2D tensor of size B x C where the labels that needs to be avoided have value 0
        :return: the misclassification loss value.
        """
        negative_rows = torch.amin(target, dim=tuple(range(1, target.dim()))) < 0

        loss = torch.zeros(negative_rows.shape, dtype=out_adv.dtype, device=out_adv.device)
        if (~negative_rows).any():
            # broadcast to a similar shape both label and prediction

            targeted_loss = self.config.crossentropy(out_adv[~negative_rows], target.abs()[~negative_rows])
            loss[~negative_rows] = -targeted_loss
        if negative_rows.any():
            # broadcast to a similar shape both label and prediction
            if self.config.losstype == 0:
                untargeted_loss = -self.config.crossentropy(
                    out_adv[negative_rows], target.abs()[negative_rows]
                )
            elif self.config.losstype == 1:
                replacement = 1 - out_adv[negative_rows].softmax(-1)
                untargeted_loss = torch.where(target == 1, replacement, 1).clamp(self.config.toll).log()
                untargeted_loss = -untargeted_loss.sum(dim=-1)
            elif self.config.losstype == 2:
                max_v, _ = out_adv[negative_rows].masked_fill(target != 0, -float("inf")).max(dim=-1)
                min_v, _ = out_adv[negative_rows].masked_fill(target == 0, float("inf")).min(dim=-1)
                untargeted_loss = -torch.log(torch.sigmoid(min_v - max_v).clamp(self.config.toll))
            else:
                raise ValueError(f"The value of loss type, {self.config.config.losstype}, does not belong in [0,1,2]")

            loss[negative_rows] = untargeted_loss

        return self.reduce(loss)
