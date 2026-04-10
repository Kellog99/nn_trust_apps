import torch
from pydantic import Field

from nn_trust import Task
from nn_trust.evaluation.statistic_factory import StatisticsFactory
from nn_trust.evaluation._statistics import Statistic, StatisticConfig


class ExpectedCalibrationErrorConfig(StatisticConfig):
    num_bins: int = Field(
        default=100,
        description="Number of bins into which the confidence scores are partitioned.",
        ge=1,
        title="Bins' number"
    ),
    original: bool = Field(
        default=False,
        description="Flag to select the version of ECE computation. If True, computes the original formulation using bin-wise mean confidence.",
        title="Original Version"
    )


@StatisticsFactory.register(
    name="Expected Calibration Error (ECE)",
    description="It is a scalar metric that quantifies the discrepancy between the predicted confidence of a classifier and the actual accuracy over a set of predictions.",
    actions={"performance"},
    task={Task.Classification}
)
class ExpectedCalibrationError(Statistic):
    r"""
    Computes the Expected Calibration Error (ECE) for a classification model.

    Expected Calibration Error is a scalar metric that quantifies the discrepancy between
    the predicted confidence of a classifier and the actual accuracy over a set of predictions.

    More formally, let the prediction set be partitioned into K bins (B₁, B₂, ..., B_K)
    based on the predicted confidence scores. For each bin Bₖ, define:

        - \(\text{acc}(B_k) = \frac{1}{|B_k|} \sum_{i \in B_k} \mathbf{1}\{ \hat{y}_i = y_i \}\),
        - \(\text{conf}(B_k) = \frac{1}{|B_k|} \sum_{i \in B_k} \hat{p}_i\),

    where \(\hat{p}_i\) is the predicted confidence for sample \(i\) and \(\mathbf{1}\{\cdot\}\) is the indicator function.

    The ECE is then given by:

        \[
        \text{ECE} = \sum_{k=1}^{K} \frac{|B_k|}{N} \left| \text{acc}(B_k) - \text{conf}(B_k) \right|,
        \]

    with \(N\) being the total number of samples.

    The parameter `original` controls whether the error is computed using the original aggregation (averaging confidence in the bin before computing the absolute difference) or an element-wise version.
    """

    CONFIG_T = ExpectedCalibrationErrorConfig

    def __init__(self, config: ExpectedCalibrationErrorConfig):
        r"""

        Postconditions:
            - Initializes empty lists to accumulate accuracy and confidence values.
            - If `num_classes` is provided, initializes `bins` as a tensor with linearly spaced values
              between \( \frac{1}{\text{num_classes}} \) and 1, partitioning the interval into `num_bins + 1` points.
        """
        super().__init__(config)

        # Accumulates binary accuracy indicators (1 for correct, 0 for incorrect).
        self.add_state("accuracy", [])
        # Accumulates predicted confidence scores.
        self.add_state("confidence", [])

        if self.config.num_classes is not None:
            # Create bins between 1/num_classes and 1 (inclusive) with num_bins+1 points.
            self.config.bins = torch.linspace(1 / self.config.num_classes, 1, self.config.num_bins + 1).to(
                self.config.device)
        else:
            self.config.bins = None

    @torch.no_grad()
    def update(self,
               y: torch.Tensor = None,
               out: torch.Tensor = None,
               **kwargs) -> None:
        r"""
        Updates the internal state.
        """
        if (y is not None) and (out is not None):
            # Determine confidence and predicted class per sample.
            confidence, y_hat = out.max(dim=-1)
            if self.config.bins is None:
                # it means that num_classes is None hence
                self.config.num_classes = out.shape[-1]

                self.config.bins = torch.linspace(
                    start=1 / self.config.num_classes,
                    end=1,
                    steps=self.config.num_bins + 1
                ).to(self.config.device)

            self.accuracy.append((y_hat == y).float())
            self.confidence.append(confidence)

    @torch.no_grad()
    def compute(self) -> float:
        r"""
        Computes the Expected Calibration Error (ECE) based on the accumulated predictions.

        Returns:
            float: The computed Expected Calibration Error.
        """
        # Convert accumulated lists to tensors.
        confidence = torch.concat(self.confidence, dim=0)
        accuracy = torch.concat(self.accuracy, dim=0)
        N = len(confidence)

        # Determine the bin indices for each confidence score.
        buckets = torch.bucketize(confidence, self.config.bins)
        bins, counts = torch.unique(buckets, return_counts=True)

        out = torch.zeros(1, device=self.config.device)
        for i in range(bins.shape[0]):
            # Compute the average accuracy for samples in the i-th bin.
            acc_bin = accuracy[buckets == bins[i]].mean()
            conf_bin = confidence[buckets == bins[i]]

            if self.config.original:
                # Compute the mean confidence for the bin and then compute the weighted absolute difference.
                conf_bin = conf_bin.mean()
                out += (acc_bin - conf_bin).abs() * counts[i]
            else:
                # Sum the absolute differences element-wise for all samples in the bin.
                out += (conf_bin - acc_bin).abs().sum()
        return (out / N).item()
