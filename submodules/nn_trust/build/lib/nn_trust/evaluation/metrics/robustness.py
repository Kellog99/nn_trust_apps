import copy
from typing import Dict, Literal, Optional

import torch
from pydantic import Field
from torchmetrics.utilities import dim_zero_cat

from nn_trust import Task
from nn_trust.evaluation.statistic_factory import StatisticsFactory
from nn_trust.evaluation._statistics import Statistic, StatisticConfig


class RobustnessConfig(StatisticConfig):
    toll: float = Field(
        default=1e-6,
        description="It represents the minimum acceptable value.",
        gt=0.0,
        title="Tolerance"

    )
    reduction: Literal["mean", "min"] = Field(
        default="mean",
        description="Method to reduce multiple measurements. Either 'mean' or 'min'. Defaults to 'mean'.",
        title="Reduction"
    )

    targeted: bool = Field(
        default=False,
        description="Tells whether it is targeted or not.",
        title="Targeted"
    )


@StatisticsFactory.register(
    name="Robustness",
    description="It computes the robustness of a model :math:`N`.",
    actions={"performance", "aggregator"},
    task={Task.Classification}
)
class Robustness(Statistic):
    r"""It computes the robustness of a model :math:`N`.
    The definition of robustness is defined for a classifier model:
        - Targeted
            .. math::
                D(x,l) \coloneqq \min\{\|x_{adv}-x\| \mid N(x_{\text{adv}}) = l, N(x_{\text{adv}})\ne N(x), x_{\text{adv}}\in X\}

        - Untargeted
            .. math::
                D(x) \coloneqq \min\{\|x_{\text{adv}}-x\| \mid N(x_{\text{adv}})\ne N(x), x_{\text{adv}}\in X\}
    """
    CONFIG_T = RobustnessConfig

    def __init__(self, config: RobustnessConfig) -> None:
        super().__init__(config)

        self.add_state("robustness", default=[], dist_reduce_fx="cat")  # save the entropy
        self.add_state("misclassification", default=[], dist_reduce_fx="cat")

    @torch.no_grad()
    def update(
            self,
            x: torch.Tensor,
            x_adv: torch.Tensor,
            y_pred: torch.Tensor,
            y_pred_adv: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> None:
        """Updates the metric state with a new batch of data.

        Args:
            x (torch.Tensor): Original input samples.
            x_adv (torch.Tensor): Adversarial examples generated from x.
            y_pred (torch.Tensor): Predictions on original samples.
            y_pred_adv (torch.Tensor): Predictions on adversarial samples.
            y (torch.Tensor, optional): Target classes for targeted attacks.
                Required when mode="targeted".
        """
        if self.config.targeted and y is None:
            raise ValueError("Target classes must be provided for targeted mode.")

        if x_adv.shape != x.shape:
            raise ValueError("The shape of the adversarial examples and the input must be the same.")

        dim = list(range(1, x.dim()))
        variation = (x - x_adv).norm(p=self.config.p, dim=dim)
        norm_x = x.norm(p=self.config.p, dim=dim).clamp(self.config.toll)
        self.robustness.append(variation / norm_x)
        misclassification = y_pred != y_pred_adv if self.config.targeted else y != y_pred_adv
        self.misclassification.append(misclassification)

    @torch.no_grad()
    def update_aggregate(self, metrics_state: Dict[str, torch.Tensor]):
        """This function take another Robustness metric state and update current metrics state
        aggregating current metric state and input metric state.
        :param metrics_state:
        """
        if len(self.robustness) == 0 and len(self.misclassification) == 0:
            assert isinstance(metrics_state["robustness"], list) and isinstance(
                metrics_state["misclassification"], list
            )
            # Save intermediate aggregate states as List[torch.Tensor] where tensor has size D (dataset)
            self.robustness = [dim_zero_cat(metrics_state["robustness"])]
            self.misclassification = [dim_zero_cat(metrics_state["misclassification"])]
        else:
            # support variables to store final values
            agg_robustness = torch.zeros_like(self.robustness[0])
            agg_misclassification = torch.zeros_like(self.misclassification[0])
            # Tensor variables to support computations
            self.robustness.append(dim_zero_cat(metrics_state["robustness"]))
            self.misclassification.append(dim_zero_cat(metrics_state["misclassification"]))
            robustness = torch.stack(self.robustness)
            misclassification = torch.stack(self.misclassification)
            # create case based masks
            mask_misclassified = torch.logical_and(misclassification[0], misclassification[1])
            mask_correct = torch.logical_not(mask_misclassified)
            mask_mixed = torch.logical_xor(misclassification[0], misclassification[1])
            # fill aggregate metric state
            if torch.any(mask_misclassified):
                agg_robustness[mask_misclassified] = robustness[mask_misclassified.repeat(2, 1)].min(dim=0).values
            if torch.any(mask_correct):
                agg_robustness[mask_correct] = robustness[mask_correct.repeat(2, 1)].max(dim=0).values
            if torch.any(mask_mixed):
                agg_robustness[mask_mixed] = robustness[mask_mixed.repeat(2, 1)][
                    misclassification[mask_mixed.repeat(2, 1)]
                ]
            # save back as List[torch.Tensor]
            self.robustness = [agg_robustness]
            self.misclassification = [agg_misclassification]

    @torch.no_grad()
    def compute(self) -> float:
        """Computes the robustness score for an attack.

        Returns:
            float: The computed robustness score.
        """
        out = torch.tensor(0.0, device=self.config.device)
        robustness = dim_zero_cat(self.robustness)
        misclassification = dim_zero_cat(self.misclassification)

        if torch.any(misclassification):
            if self.config.reduction == "mean":
                out += robustness[misclassification].mean()
            elif self.config.reduction == "min":
                out += robustness[misclassification].min()

        not_misclassified = torch.logical_not(misclassification)
        if torch.any(not_misclassified):
            out += not_misclassified.float().mean() * robustness[not_misclassified].max()
        return out.item()

    def get_raw_state(self):
        return copy.deepcopy(self.metric_state)
