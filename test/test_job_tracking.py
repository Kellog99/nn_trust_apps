import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from benchmarking.executor import BenchmarkExecutor
from models import BenchmarkOptionConfig, DatasetInfo, JobResult, ModelInfo
from nn_trust import AttackFactory, StatisticComposer, Task
from nn_trust.utils import Logger
from nn_trust.attack._cv import CVModelAdapter
from services.job_router import get_jobs


def _use_identity_attack_double(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep job-tracking tests independent of concrete attack implementations."""
    class Config:
        model_fields = {}

        @staticmethod
        def model_dump() -> dict:
            return {}

    class IdentityAttack:
        config = Config()
        logger = Logger()

        @staticmethod
        def generate(x: torch.Tensor, **_) -> torch.Tensor:
            return x

    monkeypatch.setattr(
        "benchmarking.utils.evaluation.AttackFactory.create",
        lambda **_: IdentityAttack(),
    )


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
    _use_identity_attack_double(monkeypatch)
    torch_model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
    model = CVModelAdapter(model=torch_model, task=Task.Classification)
    dataloader = DataLoader(TensorDataset(
        torch.rand(2, 3, 4, 4),
        torch.zeros(2, dtype=torch.long),
    ), collate_fn=lambda items: (
        [image for image, _ in items],
        torch.stack([label for _, label in items]),
    ))

    run_benchmark_module = importlib.import_module("benchmarking.run_benchmark")
    monkeypatch.setattr(run_benchmark_module, "load_model", lambda **_: model)
    monkeypatch.setattr(run_benchmark_module, "get_dataloader", lambda **_: dataloader)
    monkeypatch.setattr(run_benchmark_module, "get_transform_dataset", lambda **_: None)

    benchmark_id = "baseline-benchmark"
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
        benchmark_id=benchmark_id,
    )

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                config=SimpleNamespace(path_model_report_repo=str(tmp_path)),
            ),
        ),
    )
    jobs = get_jobs(
        request=request,
        benchmark_id=f" {benchmark_id} ",
        model_id="dummy-model",
        dataset_id="dummy-dataset",
        attacks_id=baseline_attack_ids,
    )

    print(f"Requested baseline attacks: {baseline_attack_ids}")
    print(f"Attacks included in the benchmark report: {sorted(reports[0].attacks)}")
    print(f"Jobs returned by get_jobs: {jobs}")

    assert [job.id for job in jobs] == baseline_attack_ids
    assert all(job.status == "finished" for job in jobs)
    assert all(job.progress == job.total == 2 for job in jobs)
    assert all(job.error is None for job in jobs)


def test_executor_persists_intermediate_batch_progress(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
) -> None:
    _use_identity_attack_double(monkeypatch)
    progress_updates: list[int] = []
    original_save = JobResult.save

    def recording_save(self: JobResult, path: Path | str) -> JobResult:
        if self.progress is not None:
            progress_updates.append(self.progress)
        return original_save(self, path)

    monkeypatch.setattr(JobResult, "save", recording_save)
    torch_model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
    model = CVModelAdapter(model=torch_model, task=Task.Classification)
    dataloader = DataLoader(TensorDataset(
        torch.rand(2, 3, 4, 4),
        torch.zeros(2, dtype=torch.long),
    ), batch_size=1, collate_fn=lambda items: (
        [image for image, _ in items],
        torch.stack([label for _, label in items]),
    ))

    output_path = tmp_path / "streaming" / "dummy-model" / "dummy-dataset"
    BenchmarkExecutor(benchmark_id="streaming", output_path=output_path).execute_jobs(
        model=model,
        dataloader=dataloader,
        attacks=[{"id": "identitybaseline"}],
        statistics=StatisticComposer(),
        max_saved_elements=1,
    )

    assert any(0 < progress < 2 for progress in progress_updates)

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                config=SimpleNamespace(path_model_report_repo=str(tmp_path)),
            ),
        ),
    )
    jobs = get_jobs(
        request=request,
        benchmark_id="streaming",
        model_id="dummy-model",
        dataset_id="dummy-dataset",
        attacks_id=["identitybaseline"],
    )
    assert jobs[0].status == "finished"
    assert jobs[0].progress == jobs[0].total == 2
