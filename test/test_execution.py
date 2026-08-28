import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader

from benchmarking import BenchmarkExecutor
from benchmarking.utils.evaluation import evaluate_attack
from models.reports import ReportAttackProps
from nn_trust import StatisticComposer, ModelAdapter
from test.test_attack import available_devices
from test.utils import get_dummy_cv_model, get_dummy_dataloader


def _fake_job_result(attack_id: str):
    """
    Build a stand-in for the `JobResult` returned by `ray.get(ref)`.
    It carries a valid, empty-ish result (valid for AttackMetricsProps) so
    BenchmarkExecutor can build a ReportAttackProps from it.
    """
    return SimpleNamespace(
        id=attack_id,
        error=None,
        result={},
        parameters=[],
    )


def make_ray_mock(values_by_ref: dict):
    """
    Build a MagicMock standing in for the `ray` module, parameterized by
    a dict {ref: JobResult_or_Exception} describing what ray.get(ref)
    should produce for each submitted task.
    """
    ray_mock = MagicMock()

    def remote_decorator(**_deco_kwargs):
        def wrapper(fn):
            remote_fn = MagicMock()
            counter = {"n": 0}

            def _remote(**call_kwargs):
                ref = f"ref-{counter['n']}"
                counter["n"] += 1
                return ref

            remote_fn.remote.side_effect = _remote
            return remote_fn

        return wrapper

    ray_mock.remote.side_effect = remote_decorator

    def _wait(refs, num_returns=1):
        return [refs[0]], refs[1:]

    ray_mock.wait.side_effect = _wait

    def _get(ref):
        outcome = values_by_ref[ref]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    ray_mock.get.side_effect = _get
    return ray_mock


@pytest.fixture
def model() -> ModelAdapter:
    return get_dummy_cv_model()


@pytest.fixture
def dataloader() -> DataLoader:
    return get_dummy_dataloader(num_samples=10)


ATTACKS = [
    {"id": "identitybaseline"},
    {"id": "contrastbaseline"},
    {"id": "gaussianbaseline"},
]


@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("max_saved_elements", [2, 3])
def test_evaluate_attack(
        model: ModelAdapter,
        dataloader: DataLoader,
        device: torch.device,
        max_saved_elements: int
):
    tmp_path: Path = Path(f"./test_tmp/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")

    checkpoint_path = tmp_path / "identitybaseline/log.pth"

    model.to(device)
    result = evaluate_attack(
        dataloader=dataloader,
        model=model,
        attack={"id": "identitybaseline"},
        statistics=StatisticComposer(),
        device=device,
        output_path=tmp_path,
        max_saved_elements=max_saved_elements
    )
    print(checkpoint_path)

    assert result.id == "identitybaseline"
    assert checkpoint_path.exists()
    data = torch.load(str(checkpoint_path), weights_only=False)
    print(data.keys())
    assert "original_input" in data
    assert "adversarial_input" in data
    assert len(data["original_input"]) == max_saved_elements
    assert len(data["adversarial_input"]) == max_saved_elements


@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("use_ray", [False, True])
def test_execution(
        model: ModelAdapter,
        dataloader: DataLoader,
        device: torch.device,
        use_ray: bool,
        monkeypatch: pytest.MonkeyPatch,
):
    """
    Given N attacks that all succeed, execute_jobs must return exactly N
    ReportAttackProps, keyed by attack id, each with error=None.
    """
    if use_ray:
        values_by_ref = {
            f"ref-{i}": _fake_job_result(attack["id"])
            for i, attack in enumerate(ATTACKS)
        }
        ray_mock = make_ray_mock(values_by_ref)
        monkeypatch.setattr("benchmarking.utils.execution.ray", ray_mock)

    executor = BenchmarkExecutor(
        device=device,
        use_ray=use_ray,
        verbose=False,
    )

    statistics: StatisticComposer = StatisticComposer()
    results: dict[str, ReportAttackProps] = executor.execute_jobs(
        model=model.to(device),
        dataloader=dataloader,
        attacks=ATTACKS,
        statistics=statistics,
    )

    expected_ids = [attack["id"] for attack in ATTACKS]
    assert list(results.keys()) == expected_ids
    for attack_id in expected_ids:
        assert results[attack_id].name == attack_id
        assert results[attack_id].metrics is not None
