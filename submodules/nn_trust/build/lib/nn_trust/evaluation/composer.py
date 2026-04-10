from typing import List

import torch
import torchmetrics
from pydantic import BaseModel, Field, field_validator

from nn_trust.attack.utils._utils import to_device
from nn_trust.evaluation.statistic_factory import StatisticsFactory as SF


class ConfigStatisticComposer(BaseModel):
    statistics: List[dict] | None = Field(
        default=[],
        description="List of all the statistics to use."
    )

    num_classes: int = Field(
        default=...,
        description="Number of possible classes",
        ge=1,
    )

    device: torch.device = Field(
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        description="The device used both for storing the eventual model and generating the attack.",
    )

    class Config:
        arbitrary_types_allowed = True

    @field_validator("statistics", mode="before")
    def valid_statistic_spec(cls, stat_spec_list):
        if len(stat_spec_list) == 0:
            return [{"name": statistic_name} for statistic_name in SF.get_list_classes()]
        else:
            # filter all the elements
            out = [stat_spec for stat_spec in stat_spec_list if stat_spec["name"] in SF.get_list_classes()]
            if len(out) == 0:
                raise ValueError("The list of attacks to perform is empty.")
        return out


class StatisticComposer(torchmetrics.Metric):
    """
    A module that helps to compose a cumulative local statistics that collects all the information of the attack
    """

    def __init__(self, config: ConfigStatisticComposer):
        super(StatisticComposer, self).__init__()
        self._config = config
        self.performance = {}
        for stat_spec in self._config.statistics:
            stat_name = stat_spec.pop("name")
            stat_spec["device"] = self._config.device
            if self._config.num_classes > 2:
                stat_spec["task"] = "multiclass"
                stat_spec["num_classes"] = self._config.num_classes
            else:
                stat_spec["task"] = "binary"
            self.performance[stat_name] = SF.create(class_id=stat_name, **stat_spec)
        self.aggregators = {k: v for k, v in self.performance.items() if "aggregator" in v.actions}
        self._evaluation_mode = "performance"
        self.statistics = self.performance

    def update(
            self,
            x: torch.Tensor,
            x_adv: torch.Tensor,
            y: torch.Tensor,
            y_pred: torch.Tensor,
            y_pred_adv: torch.Tensor,
            **kwargs,
    ) -> None:
        """
        This method updates all these statistics that measure the performance of the model for each attack

        Args:
            x_adv: the adversarial image
            x: the original image
            y: the original label
            y_pred: the model prediction on the x
            y_pred_adv: the model prediction on the x
        """
        ############ Input validation ############
        if (x_adv.dim() != 4) and (x_adv.shape != x.shape):
            raise ValueError("The adversarial input has not a proper shape.")
        if (y.size() != y_pred.size()) and (y.size() != y_pred_adv.size()):
            raise ValueError("The output of the network does not have a proper shape.")
        ##########################################

        ############ Statistics' input ############
        stats_args = {
            "x_adv": x_adv,
            "x": x,
            "y": y,
            "y_pred": y_pred,
            "y_pred_adv": y_pred_adv
        }
        ###########################################

        # Add new eventually elements
        if len(kwargs.keys()) > 0:
            stats_args.update(kwargs)

        # setting the right device to the input
        stats_args = to_device(stats_args, self._config.device)

        ################# updating the statistics #################
        for stat in self.statistics.values():
            stat.update(**stats_args)
        ###########################################################

    def compute(self):
        """
        The attack is complete therefore all the local and local aggregator statistics are computed
        """
        results = {}
        for statistic_name, statistic in self.statistics.items():
            results[statistic_name] = statistic.compute()
        return results

    def get_raw_state(self):
        raw_results = {}
        for statistic_name, statistic in self.statistics.items():

            if "aggregator" in statistic.actions:
                raw_results[statistic_name] = to_device(statistic.get_raw_state(), "cpu")
        return raw_results

    def reset(self):
        """Clean up statistic (metrics) states both standard and raw state where applicable"""
        for statistic_name, statistic in self.statistics.items():
            statistic.reset()

    def update_aggregate(self, metrics_state: dict):
        """
        The method is in charge of update internal metric state, from multiple states of the same
        type of metric.
        ITs effect is to obtain a new internal state representing a valid metric instance, that
        can produce a valid result using the standard `.compute()` method.

        :param metrics_state:  {<metric_name>: <metric_internal_state>}
        """
        for statistic_name, statistic in self.aggregators.items():
            try:
                statistic.update_aggregate(metrics_state[statistic_name])
            except KeyError:
                raise KeyError(f"The key {statistic_name} not in the metric state. Which was expected.")

    def performance(self):
        """
        Set Evaluator class in performance mode, managing statistics of performance type.
        """
        self._evaluation_mode = "performance"
        self.statistics = self.performance

    def aggregator(self):
        """
        Set Evaluator class in aggregator mode, managing statistics of aggregator type.
        """
        self._evaluation_mode = "aggregator"
        self.statistics = self.aggregators
