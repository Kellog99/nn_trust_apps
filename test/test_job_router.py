import json
from pathlib import Path

import pytest

from benchmarking import run_benchmark
from models import BenchmarkExecutionConfig, ModelReportProps, BenchmarkOptionConfig
from models import DatasetInfo, ModelInfo


@pytest.fixture
def body() -> BenchmarkExecutionConfig:
    with open("./test/utils/benchmark-request.json", "r") as f:
        data = json.load(f)
    return BenchmarkExecutionConfig.model_validate(data)


def test_start_benchmark_job(body: BenchmarkExecutionConfig, tmp_path: Path):
    """The web client's ID-only benchmark body starts with repository objects."""
    dataset: DatasetInfo = body.dataset
    model: ModelInfo = body.model

    # run_benchmark consumes serializable mappings, not the API metadata
    # model returned by /info/attacks and /info/metrics.
    attacks = [attack.model_dump(exclude_none=True) for attack in body.attacks]
    metrics = []
    for metric in body.metrics:
        metrics.append(
            {
                "id": metric.id,
                **{param.id: param.default for param in metric.parameters},
            }
        )

    # The request fixture describes the complete ImageNet catalogue, but this
    # is a router integration test, not a full-dataset benchmark.  Keep its
    # representative set large enough for neighbourhood-based metrics while
    # bounding the test's data and per-metric perturbation workload.
    dataset = dataset.model_copy(update={"batch_size": 1, "num_workers": 0})
    options: BenchmarkOptionConfig = body.options.model_copy(
        update={
            "subset": 2,
            "max_saved_elements": 1,
            "output_path": str(tmp_path),
        }
    )
    result: ModelReportProps = run_benchmark(
        models=[model],
        datasets=[dataset],
        attacks=attacks,
        metrics=metrics,
        options=options
    )[0]

    expected_attack_ids = {attack["id"] for attack in attacks} - {"identitybaseline"}
    assert set(result.attacks) == expected_attack_ids

    requested_metric_ids = {metric["id"] for metric in metrics}
    returned_metric_ids = {
        metric
        for metric, value in result.metrics.model_dump().items()
        if value is not None and metric != "num_samples"
    }
    missing_metric_ids = requested_metric_ids - returned_metric_ids
    if len(missing_metric_ids) > 0:
        print(f"Requested metrics not returned: {sorted(missing_metric_ids)}")
    assert returned_metric_ids == requested_metric_ids
    assert result.metrics.num_samples == options.subset
