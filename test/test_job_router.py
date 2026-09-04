import json

import pytest

from benchmarking import run_benchmark
from models import BenchmarkExecutionConfig, ModelReportProps, BenchmarkOptionConfig
from models import DatasetInfo, ModelInfo


@pytest.fixture
def body() -> BenchmarkExecutionConfig:
    with open("./test/utils/benchmark-request.json", "r") as f:
        data = json.load(f)
    return BenchmarkExecutionConfig.model_validate(data)


def test_start_benchmark_job(body: BenchmarkExecutionConfig):
    """The web client's ID-only benchmark body starts with repository objects."""
    dataset: DatasetInfo = body.dataset
    model: ModelInfo = body.model

    # run_benchmark consumes serializable mappings, not the API metadata
    # model returned by /info/attacks and /info/metrics.
    attacks = [attack.model_dump(exclude_none=True) for attack in body.attacks]
    metrics = [metric.model_dump(exclude_none=True) for metric in body.metrics]
    options: BenchmarkOptionConfig = body.options
    print(options.model_dump())

    result: ModelReportProps = run_benchmark(
        models=[model],
        datasets=[dataset],
        attacks=attacks,
        metrics=metrics,
        options=options
    )[0]

    assert len(result.attacks) == len(attacks)
    metric_id = [metric.id for metric in body.metrics]
    metric_not_non = [metric for metric, value in result.metrics.model_dump().items() if value is not None]
    assert len(metric_id) == len(metrics)
    assert all(m in metric_id for m in metric_not_non)
