import torch
from pydantic import Field

from nn_trust import Task
from nn_trust.evaluation._statistics import Statistic, StatisticConfig
from nn_trust.evaluation.statistic_factory import StatisticsFactory


class LipschitzBoundConfig(StatisticConfig):
    toll: float = Field(
        default=1e-4,
        description="It represents the minimum acceptable value.",
        gt=0.0,
        title="Tolerance"
    )


@StatisticsFactory.register(
    name="Lipschitz",
    description="It computes the confusion matrix of the model.",
    actions={"aggregator"},
    task={Task.Classification, Task.Detection, Task.Segmentation}
)
class LipschitzBound(Statistic):

    def __init__(self, config: LipschitzBoundConfig):
        super().__init__(config)
        # loss variable
        self.add_state("dx", default=[], dist_reduce_fx="sum")
        self.add_state("dy", default=[], dist_reduce_fx="sum")
        self.add_state("dy_dx", default=torch.tensor(0.0), dist_reduce_fx="max")
        self.add_state("L", default=torch.tensor(0.0), dist_reduce_fx="max")
        self.add_state("L_atk", default=torch.tensor(0.0), dist_reduce_fx="max")

    @torch.no_grad()
    def update(
            self,
            x: torch.Tensor = None,
            x_adv: torch.Tensor = None,
            out: torch.Tensor = None,
            out_adv: torch.Tensor = None,
            **kwargs,
    ) -> None:
        r"""This function tries to compute the Lipschitz constant, K>0, i.e.

        ... math::
            \|f(x)-f(y)\|\le K\|x-y\|.

        The idea is to use the data, the normal data and the adversarial one (if they are provided).
        :param x: original input.
        :param x_adv: adversarial input.
        :param out: original output associated to `x`.
        :param out_adv: adversarial output associated to `x_adv`.
        """

        if x is not None:
            dx = torch.norm((x - x_adv).flatten(1), p=self.config.p)
            dy = torch.norm((out - out_adv).flatten(1), p=self.config.p)
            self.L_atk = torch.max(self.L_atk, dy / dx.clamp(self.config.toll))

    def update_global_state(self) -> None:
        self.L = torch.max(self.L, self.L_atk)

    def reset_local(self) -> None:
        self.L_atk = torch.tensor(0.0, device=self.config.device)

    @torch.no_grad()
    def compute(self) -> float:
        """
        Returns the Lipschitz constant for the attack.
        """
        out = self.L_atk.item()
        self.update_global_state()
        self.reset_local()
        return out

    def compute_global_state(self) -> float:
        """
        Returns the Lipschitz constant from the global state.
        """
        return self.L.item()
