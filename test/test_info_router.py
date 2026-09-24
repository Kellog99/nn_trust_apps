from nn_trust import StatisticsFactory as SF
from nn_trust.evaluation.statistic_factory import InfoStatistic

from services.info_router import get_statistics_info


def test_statistics_info_uses_registered_tasks() -> None:
    statistics = get_statistics_info(excluded_statistics=[])

    for statistic_id, registered_object in statistics.items():
        info = InfoStatistic.model_validate(
            SF.get_information(id=statistic_id, exclude=set())
        )
        assert set(registered_object.task) == {task.name for task in info.task}
