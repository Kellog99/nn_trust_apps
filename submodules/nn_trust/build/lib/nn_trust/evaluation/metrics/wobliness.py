from math import log

import torch
from pydantic import Field

from nn_trust.attack.sampling_utils import BallSampler
from nn_trust.core import ModelAdapter, Task
from nn_trust.evaluation._statistics import Statistic, StatisticConfig
from nn_trust.evaluation.statistic_factory import StatisticsFactory


class WobblinessConfig(StatisticConfig):
    model: ModelAdapter = Field(
        default=...,
        description="Model on which it is wanted to compute the wobbliness.",
        title="Model"
    )
    samples: torch.Tensor = Field(
        default=None,
        description="Samples",
        title="Samples"
    )
    num_samples: int = Field(
        default=300,
        description="Number of points to sample.",
        ge=1,
        title="Number of samples"
    )
    ball_sample: bool = Field(
        default=True,
        description="It tells that the type of sampler is the uniform sample in the ball with radius `radius`.",
        title="Ball sampler"
    ),
    radius: float = Field(
        default=2.5,
        description="Radius of the Ball Sampler",
        gt=0.0,
        title="Ball's radius"
    )


@StatisticsFactory.register(
    name="Wobbliness",
    description="It computes the Wobbliness metric.",
    actions={"performance"},
    task={Task.Classification}
)
class Wobbliness(Statistic):
    r"""
    It computes the Expected entropy of the classes, i.e.
        ... math::
            H(x)= -\sum_{c \in C} p_c(x) \log(p_c(x))
    Then, the Wobbliness is defined as
        ... math::
            W := E[H(X)] \sim \frac{1}{N} \sum_{i=1}^N H(x_i)
    The maximum entropy is given by H_{\max} = \log(|C|),  hence
        ... math::
            \frac{W}{H_{\max}} \in [0,1]
    """

    def __init__(self, config: WobblinessConfig):
        super().__init__(config)

        self.max_entropy = log(self.config.num_classes) if self.config.num_classes is not None else 0

        self.update_metric = True
        self.add_state("entropy", default=torch.tensor(0.0), dist_reduce_fx="sum")  # save the entropy
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")  # save the number of element computed

    @torch.no_grad()
    def _set_sample(self, shape: torch.Size):
        """
        This method allows to create a sample of size `num_samples` with two possible distribution:
        1. Uniform distribution over the hyper-dimensional ball
        2. Uniform distribution over the hyper-dimensional cube
        """
        if self.config.ball_sample:
            ball = BallSampler(shape=shape, radius=self.config.radius * shape.numel() ** 0.5)
            samples = ball.sample(n=self.config.num_samples)

        else:
            samples = (torch.rand(self.config.num_samples, *shape) - 0.5) * 2 * self.config.radius

        self.config.samples = samples.to(self.config.device)

    @torch.no_grad()
    def update(self, x: torch.Tensor = None, **kwargs) -> None:
        """
        :param x: the input on which it is needed to compute the wobbliness
        """
        if x is not None:
            if self.config.samples is None:
                self._set_sample(shape=x[0].shape)
            else:
                if self.config.samples[0].shape != x[0].shape:
                    raise ValueError("The samples that were generated have a different shape.")

            ##### initialization ####
            x = x.to(self.config.device)
            sample_evaluation = self.config.model(x)
            # Check the num_classes is consistent with respect to the model output.
            if self.num_classes is None or self.num_classes != sample_evaluation.shape[-1]:
                self.num_classes = sample_evaluation.shape[-1]
                self.max_entropy = log(self.num_classes)

            # Due to a possible high number of perturbations
            # it is not feasible to compute all the xs and perturbation at the same time
            # Since the final dimensionality is B x num_samples this number can be too high
            for i in range(x.shape[0]):
                tmp = x[i].unsqueeze(0).to(self.config.device) + self.config.samples
                out = self.config.model(tmp).argmax(dim=-1)
                # Allows to compute how many times a certain integer is encountered
                # `minlength` tell the minimum number of bins that can take into consideration
                # `minlength` = `num_classes`
                classes = torch.bincount(out, minlength=self.num_classes) / self.config.num_samples
                tmp_w = -classes[classes > 0] * torch.log(classes[classes > 0])
                self.entropy += tmp_w.sum()
            self.total += x.shape[0]

    @torch.no_grad()
    def compute(self) -> float:
        denominator = self.total * self.max_entropy
        return (self.entropy / denominator).item()
