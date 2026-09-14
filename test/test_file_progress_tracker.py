import json
from pathlib import Path
from types import SimpleNamespace

from benchmarking.utils.progress_tracker import ProgressTracker
from services.job_router import get_jobs


def test_progress_is_stored_in_the_attack_directory(tmp_path: Path) -> None:
    tracker = ProgressTracker()
    benchmark_path = tmp_path / "benchmark-1"

    task_id = tracker.create_task(
        attack_id="example-attack",
        benchmark_id="benchmark-1",
        name="Example attack",
        output_path=benchmark_path,
    )
    tracker.update_task(task_id, status="in_progress", progress=42)

    status_path = benchmark_path / "example-attack" / "status.json"
    assert json.loads(status_path.read_text()) == {
        "benchmark_id": "benchmark-1",
        "attack_id": "example-attack",
        "name": "Example attack",
        "status": "in_progress",
        "progress": 42,
        "error": None,
    }


def test_get_jobs_reads_status_after_tracker_restart(
        monkeypatch,
        tmp_path: Path,
) -> None:
    benchmark_path = tmp_path / "benchmark-1"
    tracker = ProgressTracker()
    task_id = tracker.create_task(
        attack_id="broken-attack",
        benchmark_id="benchmark-1",
        output_path=benchmark_path,
    )
    tracker.update_task(
        task_id,
        status="failed",
        progress=100,
        error="example failure",
    )

    # A new tracker has no in-memory knowledge of the previous process.
    monkeypatch.setattr("benchmarking.executor.tracker", ProgressTracker())
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                config=SimpleNamespace(path_model_report_repo=str(tmp_path)),
            ),
        ),
    )

    assert get_jobs(id="benchmark-1", request=request) == [{
        "id": "broken-attack",
        "name": "broken-attack",
        "status": "failed",
        "progress": 100,
        "error": "example failure",
    }]
