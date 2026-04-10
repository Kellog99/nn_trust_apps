from typing import Literal

import torch
from pydantic import Field

from nn_trust import Task
from nn_trust.loss._loss import Loss, LossConfig
from nn_trust.loss.loss_factory import LossFactory


class SimilarityLossConfig(LossConfig):
    metric: Literal["cosine", "norm", "frob"] = Field(
        default="norm",
        description="Metric to use for the Similarity loss.",
        title="Metric"
    )


@LossFactory.register(
    name="Similarity Loss",
    description="TheSimilarity loss, in the context of Adversarial ML, make two images as similar as possible, without the need of them to be the same image.",
    task={Task.Classification, Task.Segmentation, Task.Detection}
)
class SimilarityLoss(Loss):
    r"""
    Similarity loss, in the context of Adversarial ML, make two images as similar as possible, without the need of them
    to be the same image.

    The cosine similarity distance is described in the first formula. The mse one, instead, in the second. The one to
    use can be chosen with the parameters.

    .. math::

        L_{sim} = -\left(\frac{\sum_{i,j}{(P_{i,j}N_{i,j})}}
        {\sqrt{\sum_{i,j}{P_{i,j}^{2}}}\sqrt{\sum_{i,j}{N_{i,j}^{2}}}}\right)^2 \\
        L_{sim} = \frac{1}{n}{\sum_{i, j}{(P_{i,j} - N_{i, j})^2}}

    cosine version: A. Guesmi, et al., "DAP: A Dynamic Adversarial Patch for Evading Person Detectors," 2023.
    mse version: A. Guesmi, et al., "AdvART: Adversarial Art for Camouflaged Object Detection Attacks," 2024.
    """

    CONFIG_T = SimilarityLossConfig

    def forward(self, x_adv: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Compute the similarity value between two image (of the same shape).

        :param x_adv: The adversarial image
        :param x: The first image.

        :return: the similarity value.

        \frac{\langle x, y\rangle}{\|x\|*\|y\|}
        """
        if self.config.metric == "cosine":
            x_norm = x_adv.norm(p=2, dim=(1, 2, 3), keepdim=True)
            y_norm = x_adv.norm(p=2, dim=(1, 2, 3), keepdim=True)
            cosine_similarity = torch.sum(x_adv * x) / (x_norm * y_norm)
            loss = -(cosine_similarity ** 2)
        elif self.config.metric == "norm":
            loss = (x_adv - x).norm(p=self.config.p, dim=(1, 2, 3))
        elif self.config.metric == "frob":
            loss = torch.norm(x_adv - x, p="fro", dim=(-2, -1))
        else:
            raise ValueError(f"{self.config.metric} similarity metric is not available.")

        return self.reduce(loss)
