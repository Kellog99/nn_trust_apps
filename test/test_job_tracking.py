import importlib
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from benchmarking.executor import BenchmarkExecutor, ProgressTracker
from models import BenchmarkOptionConfig, DatasetInfo, ModelInfo
from nn_trust import AttackFactory, StatisticComposer, Task
from nn_trust.attack._cv import CVModelAdapter
from services.job_router import get_jobs


@pytest.fixture
def baseline_attack_ids() -> list[str]:
    attack_ids = sorted(
        attack_id
        for attack_id in AttackFactory.get_list_classes()
        if "baseline" in attack_id
    )
    assert attack_ids == [
        "contrastbaseline",
        "gaussianbaseline",
        "identitybaseline",
        "saltnpeppernoisebaseline",
        "uniformbaseline",
    ]
    return attack_ids


def test_get_jobs_reports_all_baseline_attacks(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        baseline_attack_ids: list[str],
) -> None:
    """A completed benchmark leaves one visible job for every baseline attack."""
    progress_tracker = ProgressTracker()
    monkeypatch.setattr("benchmarking.executor.tracker", progress_tracker)

    torch_model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
    model = CVModelAdapter(model=torch_model, task=Task.Classification)
    dataloader = DataLoader(TensorDataset(
        torch.rand(2, 3, 4, 4),
        torch.zeros(2, dtype=torch.long),
    ))

    run_benchmark_module = importlib.import_module("benchmarking.run_benchmark")
    monkeypatch.setattr(run_benchmark_module, "load_model", lambda **_: model)
    monkeypatch.setattr(run_benchmark_module, "get_dataloader", lambda **_: dataloader)
    monkeypatch.setattr(run_benchmark_module, "get_transformation", lambda **_: None)

    reports = run_benchmark_module.run_benchmark(
        models=[ModelInfo(
            id="dummy-model",
            name="Dummy model",
            task="classification",
            num_classes=2,
            input_dimensionality=[3, 4, 4],
            repository=str(tmp_path),
        )],
        datasets=[DatasetInfo(
            id="dummy-dataset",
            name="Dummy dataset",
            task="classification",
            num_classes=2,
            input_dimensionality=[3, 4, 4],
            repository=str(tmp_path),
            batch_size=2,
            num_workers=0,
        )],
        attacks=[{"id": attack_id} for attack_id in baseline_attack_ids],
        metrics=[{"id": "accuracy"}],
        options=BenchmarkOptionConfig(
            gpu=False,
            max_saved_elements=1,
            output_path=str(tmp_path),
            use_ray=False,
        ),
    )

    all_jobs = get_jobs(id=None)
    benchmark_ids = {task["benchmark_id"] for task in all_jobs.values()}
    assert len(benchmark_ids) == 1
    benchmark_id = benchmark_ids.pop()
    jobs = get_jobs(id=f" {benchmark_id} ")

    print(f"Requested baseline attacks: {baseline_attack_ids}")
    print(f"Attacks included in the benchmark report: {sorted(reports[0].attacks)}")
    print(f"Jobs returned by get_jobs: {jobs}")

    assert [job["id"] for job in jobs] == baseline_attack_ids
    assert [task["attack_id"] for task in all_jobs.values()] == baseline_attack_ids
    assert all(job["status"] == "completed" for job in jobs)
    assert all(job["progress"] == 100 for job in jobs)
    assert all(job["error"] is None for job in jobs)


def test_executor_publishes_intermediate_batch_progress(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
) -> None:
    class RecordingTracker(ProgressTracker):
        def __init__(self) -> None:
            super().__init__()
            self.progress_updates: list[int] = []

        def update_task(self, task_id, status, progress=None, error=None) -> None:
            super().update_task(task_id, status, progress, error)
            if progress is not None:
                self.progress_updates.append(progress)

    progress_tracker = RecordingTracker()
    monkeypatch.setattr("benchmarking.executor.tracker", progress_tracker)
    torch_model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
    model = CVModelAdapter(model=torch_model, task=Task.Classification)
    dataloader = DataLoader(TensorDataset(
        torch.rand(2, 3, 4, 4),
        torch.zeros(2, dtype=torch.long),
    ), batch_size=1)

    BenchmarkExecutor(
        benchmark_id="streaming",
        output_path=tmp_path,
        tracker=progress_tracker,
    ).execute_jobs(
        model=model,
        dataloader=dataloader,
        attacks=[{"id": "identitybaseline"}],
        statistics=StatisticComposer(),
        max_saved_elements=1,
    )

    assert any(0 < progress < 100 for progress in progress_tracker.progress_updates)
    assert get_jobs(id="streaming")[0]["progress"] == 100
