import json
import importlib
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from fastapi import BackgroundTasks, FastAPI
from fastapi.testclient import TestClient

from benchmarking import run_benchmark
from models import BenchmarkExecutionConfig, ModelReportProps, BenchmarkOptionConfig
from models import DatasetInfo, ModelInfo

job_router = importlib.import_module("services.job_router")


@pytest.fixture
def body(tmp_path: Path) -> BenchmarkExecutionConfig:
    """Use small local repositories so the integration test is self-contained."""
    model_path = tmp_path / "model"
    model_path.mkdir()
    torch.save(torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 2)),
               model_path / "model.pth")
    dataset_path = tmp_path / "dataset"
    for label, color in enumerate(("red", "blue")):
        folder = dataset_path / str(label)
        folder.mkdir(parents=True)
        Image.new("RGB", (8, 8), color).save(folder / "image.png")
    return BenchmarkExecutionConfig.model_validate({
        "model": {"id": "test-model", "name": "Test model", "task": "classification",
                  "input_dimensionality": [3, 8, 8], "num_classes": 2,
                  "repository": str(model_path), "model_type": "plain",
                  "transformation": {"mean": [0, 0, 0], "std": [1, 1, 1], "size": 8}},
        "dataset": {"id": "test-dataset", "name": "Test dataset", "task": "classification",
                    "input_dimensionality": [3, 8, 8], "num_classes": 2,
                    "repository": str(dataset_path), "batch_size": 2, "num_workers": 0},
        "attacks": [{"id": "contrastbaseline", "name": "Contrast", "task": "classification", "parameters": []}],
        "metrics": [{"id": "accuracy", "name": "Accuracy", "task": "classification", "parameters": []}],
        "options": {"gpu": False, "verbose": False, "output_path": str(tmp_path / "reports")},
    })


def test_start_benchmark_job_returns_id_used_by_benchmark(
        body: BenchmarkExecutionConfig,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
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
                config=SimpleNamespace(
                    excluded_attacks=[],
                    path_model_report_repo=str(tmp_path),
                ),
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
        tmp_path: Path,
) -> None:
    generated_id = "20260914T120000_000001"
    monkeypatch.setattr(job_router, "create_benchmark_id", lambda: generated_id)
    monkeypatch.setattr(job_router, "run_benchmark", lambda **_: None)

    app = FastAPI()
    app.include_router(job_router.router)
    app.state.config = SimpleNamespace(
        excluded_attacks=[],
        path_model_report_repo=str(tmp_path),
    )

    response = TestClient(app).post(
        "/job/start_benchmark",
        json=body.model_dump(mode="json"),
    )

    assert response.status_code == 200
    assert response.json() == generated_id


@pytest.mark.parametrize("status", [None, "pending", "in progress", "finished", "error"])
def test_start_benchmark_records_background_setup_failures(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        status: str | None,
) -> None:
    generated_id = "20260914T120000_000001"

    def fail_benchmark(**_) -> None:
        raise RuntimeError("images in the batch have different sizes")

    monkeypatch.setattr(job_router, "run_benchmark", fail_benchmark)
    benchmark_folder = tmp_path / generated_id / "model-1" / "dataset-1"
    attack_ids = {"attack-1", "identitybaseline"}
    for attack_id in attack_ids:
        (benchmark_folder / attack_id).mkdir(parents=True)

    benchmark = BenchmarkExecutionConfig.model_validate({
        "model": {"id": "model-1", "name": "Model", "task": "classification",
                  "input_dimensionality": [3, 32, 32]},
        "dataset": {"id": "dataset-1", "name": "Dataset", "task": "classification",
                    "input_dimensionality": [3, 32, 32]},
        "attacks": [{"id": "attack-1", "name": "Attack", "task": "classification",
                     "parameters": []}],
        "metrics": [],
    })
    result_file = benchmark_folder / "attack-1" / "job_results.json"
    existing = {"id": "attack-1", "status": status, "result": {}, "error": "original error"}
    if status is not None:
        result_file.write_text(json.dumps(existing))

    job_router._run_benchmark_background(
        benchmark=benchmark,
        benchmark_folder=benchmark_folder,
        benchmark_id=generated_id,
    )

    for attack_id in attack_ids:
        result = json.loads(
            (benchmark_folder / attack_id / "job_results.json").read_text()
        )
        if attack_id == "attack-1" and status in {"finished", "error"}:
            assert result == existing
            continue
        assert result["status"] == "error"
        assert result["error"] == (
            "RuntimeError: images in the batch have different sizes"
        )


def test_start_benchmark_generates_id_when_frontend_omits_it(
        body: BenchmarkExecutionConfig,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
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
    app.state.config = SimpleNamespace(
        excluded_attacks=[],
        path_model_report_repo=str(tmp_path),
    )
    payload = body.model_dump(mode="json")
    del payload["benchmark_id"]

    response = TestClient(app).post("/job/start_benchmark", json=payload)

    assert response.status_code == 200
    assert response.json() == generated_id
    assert benchmark_call["benchmark_id"] == generated_id


def test_start_benchmark_creates_attack_folders_before_scheduling(
        body: BenchmarkExecutionConfig,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
) -> None:
    generated_id = "20260914T120000_000001"
    monkeypatch.setattr(job_router, "create_benchmark_id", lambda: generated_id)
    monkeypatch.setattr(job_router, "run_benchmark", lambda **_: None)

    excluded_attack = body.attacks[0].id
    app = FastAPI()
    app.include_router(job_router.router)
    app.state.config = SimpleNamespace(
        excluded_attacks=[excluded_attack],
        path_model_report_repo=str(tmp_path),
    )

    response = TestClient(app).post(
        "/job/start_benchmark",
        json=body.model_dump(mode="json"),
    )

    assert response.status_code == 200
    benchmark_folder = tmp_path / generated_id / body.model.id / body.dataset.id
    expected_attack_ids = {
        attack.id for attack in body.attacks if attack.id != excluded_attack
    } | {"identitybaseline"}
    assert {path.name for path in benchmark_folder.iterdir()} == expected_attack_ids


def test_get_jobs_reads_all_results_for_a_benchmark(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark-1"
    for attack_id in ("attack-a", "attack-b"):
        attack_path = benchmark_path / attack_id
        attack_path.mkdir(parents=True)
        (attack_path / "job_results.json").write_text(json.dumps({
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
    benchmark_path = tmp_path / "benchmark-1" / "model-1" / "dataset-1"
    for attack_id in ("attack-a", "attack-b"):
        attack_path = benchmark_path / attack_id
        attack_path.mkdir(parents=True)
        (attack_path / "job_results.json").write_text(json.dumps({
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
            "model_id": "model-1",
            "dataset_id": "dataset-1",
            "attacks_id": "attack-a,attack-b",
        },
    )

    assert response.status_code == 200
    assert [job["id"] for job in response.json()] == ["attack-a", "attack-b"]


def test_get_report_reads_report_from_benchmark_model_and_dataset(tmp_path: Path) -> None:
    report = {
        "info": {
            "id": "model-1",
            "name": "Model 1",
            "task": "classification",
            "input_dimensionality": [3, 32, 32],
        },
        "metrics": {"accuracy": 0.9},
        "attacks": {},
    }
    report_path = tmp_path / "benchmark-1" / "model-1" / "dataset-1"
    report_path.mkdir(parents=True)
    (report_path / "report.json").write_text(json.dumps(report), encoding="utf-8")

    app = FastAPI()
    app.include_router(job_router.router)
    app.state.config = SimpleNamespace(path_model_report_repo=str(tmp_path))

    response = TestClient(app).get(
        "/job/getReport",
        params={
            "benchmark_id": "benchmark-1",
            "model_id": "model-1",
            "dataset_id": "dataset-1",
        },
    )

    assert response.status_code == 200
    assert response.json()["info"]["id"] == "model-1"
    assert response.json()["metrics"]["accuracy"] == 0.9


def test_get_report_returns_404_when_report_does_not_exist(tmp_path: Path) -> None:
    app = FastAPI()
    app.include_router(job_router.router)
    app.state.config = SimpleNamespace(path_model_report_repo=str(tmp_path))

    response = TestClient(app).get(
        "/job/getReport",
        params={
            "benchmark_id": "benchmark-1",
            "model_id": "model-1",
            "dataset_id": "dataset-1",
        },
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Report not found"}


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

    # Exercise repository loading and reporting with a bounded local dataset.
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
