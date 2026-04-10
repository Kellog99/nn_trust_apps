from typing import Literal, Optional

import torch
from pydantic import Field

from nn_trust import Task
from nn_trust.evaluation._statistics import Statistic, StatisticConfig
from nn_trust.evaluation.metrics._utils import _update_robustness, _compute_robustness
from nn_trust.evaluation.statistic_factory import StatisticsFactory


class ClassRobustnessConfig(StatisticConfig):
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
    name="Class Robustness",
    description="It computes the robustness of a model :math:`N` for each class.",
    actions={"performance", "aggregator"},
    task={Task.Classification}
)
class ClassRobustness(Statistic):
    r"""It computes the robustness of a model :math:`N`.
         The definition of robustness is defined for a classifier model:
             - Targeted
                 .. math::
                     D(x,l) \coloneqq \min\{\|x_{adv}-x\| \mid N(x_{\text{adv}}) = l, N(x_{\text{adv}})\ne N(x), x_{\text{adv}}\in X\}

             - Untargeted
                 .. math::
                     D(x) \coloneqq \min\{\|x_{\text{adv}}-x\| \mid N(x_{\text{adv}})\ne N(x), x_{\text{adv}}\in X\}
         Args:
             device (torch.device): Device to perform calculations on. Defaults to CUDA if available, otherwise CPU.
             p (float): The p-norm to use for distance calculation. Must be >= 1.0. Defaults to 2.0 (Euclidean norm).
             toll (float): Small value to prevent division by zero. Must be > 0.0. Defaults to 1e-6.
             reduction (str): Method to reduce multiple measurements. Either "mean" or "min". Defaults to "mean".
            **kwargs: Additional arguments passed to parent class.
         """

    def __init__(self, config: ClassRobustnessConfig):
        super().__init__(config)
        # for the global state
        self.add_state("robustness", default=torch.Tensor([]))  # save the entropy
        self.add_state("misclassification", default=torch.Tensor([]))

        # for the local state
        self.add_state("atk_robustness", default=[], dist_reduce_fx="cat")  # save the entropy
        self.add_state("atk_misclassification", default=[], dist_reduce_fx="cat")

        self.add_state("classes", default=[])  # save the entropy
        self.add_state("tmp_classes", default=[])  # save the entropy

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
            **kwargs: Additional keyword arguments (not used).

        Raises:
            ValueError: If mode is "targeted" but target_classes is None.
        """
        if self.config.targeted and y is None:
            raise ValueError("Target classes must be provided for targeted mode.")

        if x_adv.shape != x.shape:
            raise ValueError("The shape of the adversarial examples and the input must be the same.")

        if len(self.classes) == 0:
            self.tmp_classes.append(y_pred)

        variation = torch.norm((x - x_adv).flatten(1), p=self.config.p, dim=1)
        norm_x = torch.norm(x.flatten(1), p=self.config.p, dim=1)
        self.atk_robustness.append(variation / norm_x.clamp(self.config.toll))
        misclassification = y_pred != y_pred_adv if self.config.targeted else y != y_pred_adv
        self.atk_misclassification.append(misclassification)

    def reset_local(self):
        """Reset the local variables for a new evaluation round."""
        self.atk_robustness = []
        self.atk_misclassification = []
        self.tmp_classes = []

    def update_global_state(self) -> None:
        if len(self.classes) == 0:
            self.classes = torch.cat(self.tmp_classes)

        if len(self.robustness) == 0:
            self.robustness = torch.cat(self.atk_robustness)
            self.misclassification = torch.cat(self.atk_misclassification)

        else:
            robustness = torch.cat(self.atk_robustness)
            misclassification = torch.cat(self.atk_misclassification)
            if len(robustness) != len(self.robustness):
                raise ValueError(
                    f"The local state and the global state have different size ({len(robustness)}, {len(self.robustness)})"
                )

            misclassification, robustness = _update_robustness(
                atk_robustness=robustness,
                robustness=self.robustness,
                atk_misclassification=misclassification,
                misclassification=self.misclassification,
            )
            self.robustness = robustness
            self.misclassification = misclassification

    @torch.no_grad()
    def compute(self) -> list[float]:
        """Computes the robustness score for an attack.

        Returns:
            float: The computed robustness score.
        """
        if len(self.classes) == 0:
            self.classes = torch.cat(self.tmp_classes)

        self.update_global_state()

        atk_misclassification = torch.cat(self.atk_misclassification)
        atk_robustness = torch.cat(self.atk_robustness)

        out = torch.zeros(self.config.num_classes, device=self.config.device)
        for i in range(self.config.num_classes):
            # if none of the input belongs into a class then it is set a default value for that specific class
            if torch.any(self.classes == i):
                out[i] += _compute_robustness(
                    robustness=atk_robustness[self.classes == i],
                    misclassification=atk_misclassification[self.classes == i],
                )
        return out.tolist()

    def compute_global_state(self) -> list[float]:
        """
        Returns the robustness score calculated from the global state.
        """
        out = torch.zeros(self.config.num_classes, device=self.config.device)
        for i in range(self.config.num_classes):
            # if none of the input belongs into a class then it is set a default value for that specific class
            if torch.any(self.classes == i):
                out[i] += _compute_robustness(
                    robustness=self.robustness[self.classes == i],
                    misclassification=self.misclassification[self.classes == i],
                )
        return out.tolist()
