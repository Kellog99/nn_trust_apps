import torch
from pydantic import Field
from torchmetrics.classification import MulticlassConfusionMatrix

from nn_trust import Task
from nn_trust.evaluation.statistic_factory import StatisticsFactory
from nn_trust.evaluation._statistics import Statistic, StatisticConfig


class MeanSquareContingencyConfig(StatisticConfig):
    toll: float = Field(
        default=1e-6,
        description="It represents the minimum acceptable value.",
        gt=0.0,
        title="Tolerance"
    )


@StatisticsFactory.register(
    name="Mean Square Contingency (MSC)",
    description="It quantifies the dependency between the predicted and actual class labels using the confusion matrix. It measures the squared deviations of the observed joint probability from the product of the marginal probabilities.",
    actions={"performance"},
    task={Task.Classification}
)
class MeanSquareContingency(Statistic):
    r"""
    Computes the Mean Square Contingency (MSC) statistic for multiclass classification tasks.

    The Mean Square Contingency statistic quantifies the dependency between the predicted
    and actual class labels using the confusion matrix. It measures the squared deviations
    of the observed joint probability from the product of the marginal probabilities.

    Given a normalized confusion matrix \(P\) with entries \(p_{ij}\), where:

        \[
        p_{ij} = \frac{n_{ij}}{N}, \quad \text{with } N = \sum_{i,j} n_{ij},
        \]

    and where:

        - \(n_{ij}\) is the number of samples with true label \(i\) and predicted label \(j\),

    the marginal probabilities are computed as:

        \[
        p_i = \sum_{j} p_{ij} \quad \text{and} \quad p_j = \sum_{i} p_{ij}.
        \]

    The Mean Square Contingency is then defined as:

        \[
        \phi = \sum_{i=1}^{C}\sum_{j=1}^{C} \frac{\left( p_{ij} - p_i p_j \right)^2}{p_i p_j},
        \]

    where \(C\) is the number of classes.

    """

    def __init__(self, config: MeanSquareContingencyConfig):
        super().__init__(config)
        # loss variable
        self.confusion_matrix = MulticlassConfusionMatrix(
            num_classes=self.config.num_classes,
            normalize="all"
        ).to(self.config.device)

    @torch.no_grad()
    def update(
            self,
            y: torch.Tensor = None,
            y_pred: torch.Tensor = None,
            **kwargs
    ) -> None:
        r"""
        Updates the confusion matrix with a new batch of true labels and predictions.

        Parameters:
            y (torch.Tensor): Ground-truth labels (shape: [N]), where N is the number of samples.
            y_pred (torch.Tensor): Predicted labels or logits (shape: [N] or [N, C]).
                                   The behavior of `MulticlassConfusionMatrix.update` should handle
                                   logits or class indices accordingly.
            **kwargs: Additional keyword arguments for further customization or extension.
        """
        self.confusion_matrix.update(preds=y_pred, target=y)

    @torch.no_grad()
    def compute(self) -> float:
        r"""
        Computes the Mean Square Contingency (MSC) statistic using the accumulated confusion matrix.

        Returns:
            float: The computed Mean Square Contingency statistic.
        """
        # Compute the normalized confusion matrix.
        confusion_matrix = self.confusion_matrix.compute()

        r = confusion_matrix.sum(1).unsqueeze(1)  # sum on the column hence it is p_i
        c = confusion_matrix.sum(0).unsqueeze(0)  # sum on the rows hence it is p_j
        prod = torch.matmul(r, c)
        phi = (confusion_matrix - prod).pow(2) / prod.clamp(self.config.toll)
        return phi.sum().item()
