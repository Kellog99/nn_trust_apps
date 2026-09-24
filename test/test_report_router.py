import importlib
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from models.info import ModelInfo
from models.reports import ModelReportProps, ReportMetricsProps

report_router = importlib.import_module("services.report_router")


@pytest.mark.parametrize("extra_metrics", [
    {},
    {"imagevariance": [1.537, 1.544, 1.776]},
    {
        "imagevariance": [1.537, 1.544, 1.776],
        "custom_matrix": [[1, 2], [3, 4]],
        "custom_details": {"value": 0.5},
        "custom_label": "unavailable",
        "custom_flag": True,
        "custom_missing": None,
    },
])
def test_benchmarks_returns_only_scalar_metrics(monkeypatch, tmp_path, extra_metrics):
    report = ModelReportProps(
        info=ModelInfo(
            id="model-1", name="Test model", parameters=100,
            task="image_classification", input_dimensionality=[3, 32, 32],
        ),
        metrics=ReportMetricsProps(
            accuracy=0.75, num_samples=4, custom_score=0.6,
            confusion_matrix=[[2, 1], [0, 1]], **extra_metrics,
        ),
        attacks={},
    )
    original_metrics = report.metrics.model_dump()
    monkeypatch.setattr(report_router, "get_info", lambda **kwargs: [report])
    app = FastAPI()
    app.include_router(report_router.router)
    app.state.config = SimpleNamespace(path_model_report_repo=str(tmp_path))

    response = TestClient(app).get("/report/benchmarks")

    assert response.status_code == 200
    assert response.json() == [{
        "name": "Test model", "param": 100, "task": "image_classification",
        "benchmark_id": "model-1",
        "metrics": {"accuracy": 0.75, "num_samples": 4, "custom_score": 0.6},
    }]
    assert report.metrics.model_dump() == original_metrics
