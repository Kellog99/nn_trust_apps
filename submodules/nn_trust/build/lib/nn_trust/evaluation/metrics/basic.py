from typing import Optional

import torch
import torchmetrics
from annotated_types import Literal
from pydantic import Field

from nn_trust import Task
from nn_trust.evaluation._statistics import Statistic, StatisticConfig
from nn_trust.evaluation.statistic_factory import StatisticsFactory


@StatisticsFactory.register(
    name="Count Samples",
    description="It keeps track of number of samples used to update the statistic.",
    actions={"performance"}
)
class CountSamples(Statistic):
    def __init__(self, config: StatisticConfig = StatisticConfig):
        super().__init__(config)
        self.add_state("n_samples", torch.tensor(0, dtype=torch.long))

    def update(self, x, **kwargs) -> None:
        self.n_samples += x.shape[0]

    def compute(self) -> torch.Tensor:
        return self.n_samples.item()

    def __repr__(self):
        return "count-samples"


class BasicClassificationConfig(StatisticConfig):
    task: Literal["binary", "multiclass", "multilabel"] = Field(
        default="multiclass",
        description="Task where the statistic is used.",
        title="Task"
    )
    average_method: Literal["macro", "weighted", "micro"] = Field(
        default="micro",
        description="Type of average method to use.",
        title="Average method"
    )
    targeted: bool = Field(
        default=False,
        description="Tells whether the classification regard a targeted or an untargeted prediction.",
        title="Targeted."
    )


class AccuracyConfig(BasicClassificationConfig):
    pass


@StatisticsFactory.register(
    name="Accuracy",
    description="It computes the accuracy.",
    actions={"performance"},
    task={Task.Classification}
)
class Accuracy(Statistic):
    CONFIG_T = AccuracyConfig

    def __init__(
            self,
            config: AccuracyConfig,
    ):
        super().__init__(config)
        self.accuracy = torchmetrics.Accuracy(
            num_classes=self.config.num_classes,
            task=self.config.task,
            average=self.config.average_method
        ).to(self.config.device)

    def reset(self) -> None:
        self.accuracy.reset()

    def update(
            self,
            y_pred: torch.Tensor,
            y_pred_adv: torch.Tensor,
            **kwargs
    ):
        if y_pred.dim() != 1:
            raise ValueError("The inputs refer to the classes.")

        self.accuracy.update(preds=y_pred_adv, target=y_pred)

    def compute(self) -> float:
        return self.accuracy.compute().item()


class ConfusionMatrixConfig(BasicClassificationConfig):
    pass


@StatisticsFactory.register(
    name="Confusion Matrix",
    description="It computes the confusion matrix of the model.",
    actions={"performance"},
    task={Task.Classification}
)
class ConfusionMatrix(Statistic):
    CONFIG_T = ConfusionMatrixConfig

    def __init__(self, config: ConfusionMatrixConfig):
        super().__init__(config)
        self.confusion_matrix = torchmetrics.classification.MulticlassConfusionMatrix(
            num_classes=self.config.num_classes)
        self.confusion_matrix.to(config.device)

    @torch.no_grad()
    def update(
            self,
            y_pred: Optional[torch.Tensor] = None,
            y_pred_adv: Optional[torch.Tensor] = None,
            **kwargs
    ):
        if (y_pred_adv is not None) and (y_pred is not None):
            self.confusion_matrix.update(target=y_pred, preds=y_pred_adv)

    @torch.no_grad()
    def compute(self) -> list[list[float]]:
        return self.confusion_matrix.compute().cpu().tolist()


class F1scoreConfig(BasicClassificationConfig):
    pass


@StatisticsFactory.register(
    name="F1 score",
    description="It computes the f1 score.",
    actions={"performance"},
    task={Task.Classification}
)
class F1score(Statistic):
    CONFIG_T = F1scoreConfig

    def __init__(
            self,
            config: F1scoreConfig,
    ):
        super().__init__(config)
        self.f1_score = torchmetrics.F1Score(
            num_classes=self.config.num_classes,
            task=self.config.task,
            average=self.config.average_method
        ).to(self.config.device)

    def reset(self) -> None:
        self.f1_score.reset()

    def update(
            self,
            y_pred: torch.Tensor,
            y_pred_adv: torch.Tensor,
            **kwargs
    ):
        if y_pred.dim() != 1:
            raise ValueError("The inputs refer to the classes.")

        self.f1_score.update(preds=y_pred_adv, target=y_pred)

    def compute(self) -> float:
        return self.f1_score.compute().item()


class MisclassificationConfig(BasicClassificationConfig):
    pass


@StatisticsFactory.register(
    name="Misclassification",
    description="It computes the Misclassification.",
    actions={"performance"},
    task={Task.Classification}
)
class Misclassification(Statistic):
    CONFIG_T = MisclassificationConfig

    def __init__(
            self,
            config: MisclassificationConfig
    ):
        super().__init__(config)
        self.add_state("misclassification", default=[], dist_reduce_fx="cat")

    def update(
            self,
            y_pred_adv: torch.Tensor,
            y_pred: torch.Tensor,
            y_target: torch.Tensor = None,
            **kwargs
    ):
        if y_pred.dim() != 1:
            raise ValueError("The inputs refer to the classes.")
        if self.config.targeted:
            if y_target is None:
                raise ValueError(
                    "If the misclassification is targeted, then it is necessary to pass the targeted class."
                )
            self.misclassification.append(y_target == y_pred_adv)
        else:
            self.misclassification.append(y_pred != y_pred_adv)

    def compute(self) -> float:
        out = torch.cat(self.misclassification)
        return out.float().mean().item()

    def reset(self) -> None:
        self.misclassification = []


class PrecisionConfig(BasicClassificationConfig):
    pass


@StatisticsFactory.register(
    name="Precision",
    description="It computes the precision of the predictions.",
    actions={"performance"},
    task={Task.Classification}
)
class Precision(Statistic):
    CONFIG_T = PrecisionConfig

    def __init__(self, config: PrecisionConfig):
        super().__init__(config)
        self.precision = torchmetrics.Precision(
            num_classes=self.config.num_classes,
            task=self.config.task,
            average=self.config.average_method
        ).to(self.config.device)

    def reset(self) -> None:
        self.precision.reset()

    def update(
            self,
            y_pred: torch.Tensor,
            y_pred_adv: torch.Tensor,
            **kwargs
    ) -> None:
        if y_pred.dim() != 1:
            raise ValueError("The inputs refer to the classes.")

        self.precision.update(preds=y_pred_adv, target=y_pred)

    def compute(self) -> float:
        return self.precision.compute().item()
