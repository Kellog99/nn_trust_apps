from typing import Literal

from pydantic import Field

from nn_trust import Task
from nn_trust.evaluation._statistics import Statistic
from nn_trust.factory import Info, Factory

stat_action = Literal["performance", "aggregator"]


class InfoStatistic(Info[Statistic]):
    actions: set[stat_action] = Field(
        default=...,
        description="This set represent the domain of action of the statistic",
    )


class StatisticsFactory(Factory):
    """
    This is the register/factory class associated to the statistics
    """
    _info_type = InfoStatistic

    @classmethod
    def filter(
            cls,
            info: InfoStatistic,
            task: set[Task] | None = None,
            actions: set[stat_action] | None = None,
            **kwargs
    ) -> bool:
        """
        The filter for the statistics
        """
        in_task: bool = super().filter(info=info, task=task, **kwargs)
        in_action: bool = len(info.actions.intersection(actions)) > 0 if actions else True

        return in_task and in_action
