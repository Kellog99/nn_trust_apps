from typing import List

from nn_trust.attack.evaluation._statistics import StatisticsFactory as SF
from nn_trust.core import Task

from models import MetricProps


def get_metrics(task: Task = Task.Classification) -> List[dict]:
    """
    Return the list of all the available metrics that are computable during a benchmark.
    """

    return {
        stat:MetricProps(
            id=stat,
            name=stat,
        ).model_dump() for stat in SF.list_statistics(name=False, task=task)
    }
