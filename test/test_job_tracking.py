from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from benchmarking.executor import BenchmarkExecutor, ProgressTracker
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
    progress_tracker = ProgressTracker()
    monkeypatch.setattr("benchmarking.executor.tracker", progress_tracker)

    torch_model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
    model = CVModelAdapter(model=torch_model, task=Task.Classification)
    dataloader = DataLoader(TensorDataset(
        torch.rand(2, 3, 4, 4),
        torch.zeros(2, dtype=torch.long),
    ))

    BenchmarkExecutor(
        benchmark_id="benchmark-1",
        output_path=tmp_path,
    ).execute_jobs(
        model=model,
        dataloader=dataloader,
        attacks=[{"id": attack_id} for attack_id in baseline_attack_ids],
        statistics=StatisticComposer(),
        max_saved_elements=1,
    )

    jobs = get_jobs(id=" benchmark-1 ")
    all_jobs = get_jobs(id=None)

    assert [job["id"] for job in jobs] == baseline_attack_ids
    assert [task["attack_id"] for task in all_jobs.values()] == baseline_attack_ids
    assert all(job["status"] == "completed" for job in jobs)
    assert all(job["progress"] == 100 for job in jobs)
    assert all(job["error"] is None for job in jobs)
