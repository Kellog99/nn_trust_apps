import torch
import torchmetrics

from nn_trust import Task
from nn_trust.evaluation.statistic_factory import StatisticsFactory
from nn_trust.evaluation._statistics import Statistic, StatisticConfig


class SSIMConfig(StatisticConfig):
    pass


@StatisticsFactory.register(
    name="Structural Similarity Index Measure (SSIM)",
    description="The Structural Similarity Index (SSIM) is a perceptual metric that quantifies image quality degradation caused by processing.",
    actions={"performance"},
    task={Task.Classification}
)
class SSIM(Statistic):
    CONFIG_T = SSIMConfig

    def __init__(self, config: SSIMConfig):
        super().__init__(config)
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure().to(self.config.device)

    def reset(self) -> None:
        self.ssim.reset()

    def update(self,
               x: torch.Tensor,
               x_adv: torch.Tensor,
               **kwargs):
        if x.dim() != 4:
            raise ValueError(f"The dimensionality of the input should be BxCxHxW but it received {x.shape}.")
        self.ssim.update(preds=x_adv.to(self.config.device), target=x.to(self.config.device))

    def compute(self) -> float:
        return self.ssim.compute().item()
