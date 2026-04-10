import torch
import torchmetrics

from nn_trust import Task
from nn_trust.evaluation._statistics import StatisticConfig, Statistic
from nn_trust.evaluation.statistic_factory import StatisticsFactory


class MeanConfig(StatisticConfig):
    pass


@StatisticsFactory.register(
    name="Mean",
    description="It computes the Dataset's mean for each channel.",
    actions={"performance"},
    task={Task.Classification}
)
class ImageMean(Statistic):
    CONFIG_T = MeanConfig

    def __init__(
            self,
            config: MeanConfig,
    ):
        super().__init__(config)
        self.mean = torch.zeros(3, device=self.config.device)
        self.num_examples = 0

    def reset(self) -> None:
        self.mean = torch.zeros(3, device=self.config.device)
        self.num_examples = 0

    def update(
            self,
            x_adv: torch.Tensor,
            **kwargs
    ):
        if x_adv.dim() != 4:
            raise ValueError(f"The must be expected as a 4D tensor, BxCxHxW, but it receives {x_adv.shape}.")

        self.mean = (x_adv.mean(dim=(2, 3)).sum(0) + self.num_examples * self.mean) / (
                self.num_examples + x_adv.shape[0])
        self.num_examples += x_adv.shape[0]

    def compute(self) -> list[float]:
        return [round(m, 3) for m in self.mean.tolist()]


class VarianceConfig(StatisticConfig):
    pass


@StatisticsFactory.register(
    name="Variance",
    description="It computes the Dataset's variance for each channel.",
    actions={"performance"},
    task={Task.Classification}
)
class ImageVariance(Statistic):
    CONFIG_T = VarianceConfig

    def __init__(
            self,
            config: VarianceConfig,
    ):
        super().__init__(config)
        self.mean = torch.zeros(3, device=self.config.device)
        self.second_moment = torch.zeros(3, device=self.config.device)
        self.num_examples = 0

    def reset(self) -> None:
        self.mean = torch.zeros(3, device=self.config.device)
        self.second_moment = torch.zeros(3, device=self.config.device)
        self.num_examples = 0

    def update(
            self,
            x_adv: torch.Tensor,
            **kwargs
    ):
        if x_adv.dim() != 4:
            raise ValueError(f"The must be expected as a 4D tensor, BxCxHxW, but it receives {x_adv.shape}.")

        new_mean = x_adv.mean(dim=(2, 3))
        new_n = x_adv.shape[0]
        self.mean = (new_mean.sum(0) + self.num_examples * self.mean) / (self.num_examples + new_n)
        self.second_moment = (new_mean.pow(2).sum(0) + self.num_examples * self.second_moment) / (
                self.num_examples + new_n)
        self.num_examples += new_n

    def compute(self) -> list[float]:
        return [round(m, 3) for m in (self.second_moment - self.mean.pow(2)).tolist()]
