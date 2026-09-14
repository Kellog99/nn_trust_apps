import json
import importlib
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, FastAPI
from fastapi.testclient import TestClient

from benchmarking import run_benchmark
from models import BenchmarkExecutionConfig, ModelReportProps, BenchmarkOptionConfig
from models import DatasetInfo, ModelInfo

job_router = importlib.import_module("services.job_router")


@pytest.fixture
def body() -> BenchmarkExecutionConfig:
    with open("./test/utils/benchmark-request.json", "r") as f:
        data = json.load(f)
    return BenchmarkExecutionConfig.model_validate(data)


def test_start_benchmark_job_returns_id_used_by_benchmark(
        body: BenchmarkExecutionConfig,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_id = "20260914T120000_000001"
    monkeypatch.setattr(job_router, "create_benchmark_id", lambda: generated_id)
    benchmark_call = {}
    monkeypatch.setattr(
        job_router,
        "run_benchmark",
        lambda **kwargs: benchmark_call.update(kwargs),
    )

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                config=SimpleNamespace(excluded_attacks=[]),
            ),
        ),
    )
    background_tasks = BackgroundTasks()
    result = asyncio.run(
        job_router.start_benchmark_job(request, background_tasks, body)
    )

    assert result == generated_id
    assert benchmark_call == {}

    asyncio.run(background_tasks())

    assert benchmark_call["benchmark_id"] == generated_id


def test_start_benchmark_http_response_has_content_and_200_status(
        body: BenchmarkExecutionConfig,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_id = "20260914T120000_000001"
    monkeypatch.setattr(job_router, "create_benchmark_id", lambda: generated_id)
    monkeypatch.setattr(job_router, "run_benchmark", lambda **_: None)

    app = FastAPI()
    app.include_router(job_router.router)
    app.state.config = SimpleNamespace(excluded_attacks=[])

    response = TestClient(app).post(
        "/job/start_benchmark",
        json=body.model_dump(mode="json"),
    )

    assert response.status_code == 200
    assert response.json() == generated_id


def test_start_benchmark_generates_id_when_frontend_omits_it(
        body: BenchmarkExecutionConfig,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_call = {}
    generated_id = "20260914T120000_000001"
    monkeypatch.setattr(job_router, "create_benchmark_id", lambda: generated_id)
    monkeypatch.setattr(
        job_router,
        "run_benchmark",
        lambda **kwargs: benchmark_call.update(kwargs),
    )

    app = FastAPI()
    app.include_router(job_router.router)
    app.state.config = SimpleNamespace(excluded_attacks=[])
    payload = body.model_dump(mode="json")
    del payload["benchmark_id"]

    response = TestClient(app).post("/job/start_benchmark", json=payload)

    assert response.status_code == 200
    assert response.json() == generated_id
    assert benchmark_call["benchmark_id"] == generated_id


def test_get_jobs_reads_all_results_for_a_benchmark(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark-1"
    for attack_id in ("attack-a", "attack-b"):
        attack_path = benchmark_path / attack_id
        attack_path.mkdir(parents=True)
        (attack_path / "results.json").write_text(json.dumps({
            "id": attack_id,
            "status": "in progress",
            "progress": 1,
            "total": 2,
        }))

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                config=SimpleNamespace(path_model_report_repo=str(tmp_path)),
            ),
        ),
    )

    jobs = job_router.get_jobs(
        request=request,
        benchmark_id=" benchmark-1 ",
        attacks_id=None,
    )

    assert [job.id for job in jobs] == ["attack-a", "attack-b"]


def test_get_jobs_accepts_quoted_id_and_comma_separated_attacks(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark-1"
    for attack_id in ("attack-a", "attack-b"):
        attack_path = benchmark_path / attack_id
        attack_path.mkdir(parents=True)
        (attack_path / "results.json").write_text(json.dumps({
            "id": attack_id,
            "status": "in progress",
            "progress": 1,
            "total": 2,
        }))

    app = FastAPI()
    app.include_router(job_router.router)
    app.state.config = SimpleNamespace(path_model_report_repo=str(tmp_path))

    response = TestClient(app).get(
        "/job/getJobs",
        params={
            "benchmark_id": '"benchmark-1"',
            "attacks_id": "attack-a,attack-b",
        },
    )

    assert response.status_code == 200
    assert [job["id"] for job in response.json()] == ["attack-a", "attack-b"]


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
