import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader

from benchmarking import BenchmarkExecutor
from benchmarking.utils.evaluation import evaluate_attack
from models.reports import ReportAttackProps, AttackMetricsProps
from nn_trust import StatisticComposer, ModelAdapter
from test.test_single_attack import available_devices
from test.utils import get_dummy_cv_model, get_dummy_dataloader


def _fake_job_result(attack_id: str, error: Optional[str] = None):
    """
    Stand-in for the `JobResult` returned by `ray.get(ref)`.
    error=None => success path; error="..." => failure path.
    """
    return SimpleNamespace(
        id=attack_id,
        error=error,
        result={} if error is None else None,
        parameters=[] if error is None else None,
    )


def make_ray_mock(values_by_ref: Optional[dict] = None) -> MagicMock:
    """
    Stand-in for the `ray` module, parameterized by {ref: JobResult_or_Exception}
    describing what ray.get(ref) should produce for each submitted task.

    Supports BOTH `ray.remote` call conventions so the mock stays valid
    regardless of how `_iter_ray` invokes it:
      - direct:     ray.remote(fn)              -> remote_fn
      - decorator:  ray.remote(**kwargs)(fn)     -> remote_fn
    """
    values_by_ref = values_by_ref if values_by_ref is not None else {}
    ray_mock = MagicMock()
    counter = {"n": 0}

    def _make_remote_fn():
        remote_fn = MagicMock()

        def _remote(**_call_kwargs):
            ref = f"ref-{counter['n']}"
            counter["n"] += 1
            return ref

        remote_fn.remote.side_effect = _remote
        return remote_fn

    def _remote_dispatch(*args, **kwargs):
        if args and callable(args[0]):
            # direct form: ray.remote(fn)
            return _make_remote_fn()

        # decorator form: ray.remote(**kwargs) -> wrapper(fn)
        def wrapper(fn):
            return _make_remote_fn()

        return wrapper

    ray_mock.remote.side_effect = _remote_dispatch

    def _wait(refs, num_returns=1):
        return refs[:num_returns], refs[num_returns:]

    ray_mock.wait.side_effect = _wait

    def _get(ref):
        outcome = values_by_ref[ref]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    ray_mock.get.side_effect = _get
    ray_mock.is_initialized.return_value = True
    return ray_mock


@pytest.fixture
def model() -> ModelAdapter:
    return get_dummy_cv_model()


@pytest.fixture
def dataloader() -> DataLoader:
    return get_dummy_dataloader(num_samples=10)


@pytest.fixture
def tmp_path() -> Path:
    return Path("./tmp")


@pytest.fixture
def attacks() -> list[dict]:
    return [
        {"id": "identitybaseline"},
        {"id": "contrastbaseline"},
        {"id": "gaussianbaseline"},
    ]


@pytest.fixture
def statistics() -> list[dict]:
    return [
        {"id": "accuracy"},
        {"id": "f1score"},
        {"id": "misclassification"},
        {"id": "precision"},
    ]


# ---------------------------------------------------------------------------
# evaluate_attack
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("max_saved_elements", [2, 3, None])
def test_evaluate_attack(
        model: ModelAdapter,
        dataloader: DataLoader,
        device: torch.device,
        tmp_path: Path,
        max_saved_elements: int,
):
    mse: int = max_saved_elements or 1
    tmp_path: Path = tmp_path / mse
    tmp_path.mkdir(exist_ok=True, parents=True)
    checkpoint_path = tmp_path / "identitybaseline" / "log.pth"

    model.to(device)
    statistics = StatisticComposer()
    result = evaluate_attack(
        dataloader=dataloader,
        model=model,
        attack={"id": "identitybaseline"},
        statistics=statistics,
        device=device,
        output_path=tmp_path,
        max_saved_elements=max_saved_elements,
    )

    assert result.id == "identitybaseline", "no identity"

    assert checkpoint_path.exists(), f"the file does not exists in {checkpoint_path}"
    data = torch.load(str(checkpoint_path), weights_only=False)
    assert "original_input" in data, "No original input in data"
    assert "adversarial_input" in data, "No adversarial data in the input"

    orig: int = len(data["original_input"])
    adv: int = len(data["adversarial_input"])
    assert orig == mse, f"The original, {orig}, is not the one that is required, {mse}"
    assert adv == mse, f"The Adv, {adv}, input is not the one that is required, {mse}"


# ---------------------------------------------------------------------------
# BenchmarkExecutor.execute_jobs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("use_ray", [False, True])
def test_execution(
        model: ModelAdapter,
        dataloader: DataLoader,
        attacks: list[dict],
        device: torch.device,
        use_ray: bool,
        tmp_path: Path,
        statistics: list[dict],
        monkeypatch: Optional[pytest.MonkeyPatch],
):
    """
    Given N attacks that all succeed, execute_jobs must return exactly N
    ReportAttackProps, keyed by attack id, each with usable metrics.
    """
    if use_ray:
        values_by_ref = {
            f"ref-{i}": _fake_job_result(attack["id"])
            for i, attack in enumerate(attacks)
        }
        if monkeypatch is None:
            raise ValueError("It has to be defined.")
        monkeypatch.setattr(
            "benchmarking.utils.execution.ray",
            make_ray_mock(values_by_ref),
        )

    executor = BenchmarkExecutor(
        device=device,
        use_ray=use_ray,
        verbose=False,
        output_path=tmp_path
    )
    stat_composer = StatisticComposer(statistics=statistics)
    results: dict[str, ReportAttackProps] = executor.execute_jobs(
        model=model.to(device),
        dataloader=dataloader,
        attacks=attacks,
        statistics=stat_composer,
    )

    tmp_path.mkdir(exist_ok=True, parents=True)

    tmp = {
        key: value.model_dump()
        for key, value in results.items()
    }

    with open(tmp_path / "results.json", "w", encoding="utf-8") as f:
        json.dump(tmp, f, indent=4)

    expected_ids = [attack["id"] for attack in attacks]
    assert list(results.keys()) == expected_ids
    for attack_id in expected_ids:
        assert results[attack_id].name == attack_id
        assert results[attack_id].metrics is not None

    # for each attack I check that all the metrics that are needed exists
    for attack in attacks:
        id = attack["id"]
        metric = results[id].metrics.model_dump()
        for stat in statistics:
            assert metric[stat["id"]] is not None, f"Stat {stat['id']} is None"


if __name__ == "__main__":
    test_execution(
        model=get_dummy_cv_model(),
        dataloader=get_dummy_dataloader(num_samples=10),
        attacks=[
            {"id": "identitybaseline"},
            {"id": "contrastbaseline"},
            {"id": "gaussianbaseline"},
        ],
        device=torch.device("cpu"),
        use_ray=False,
        monkeypatch=None,
        tmp_path=Path(f"./tmp/{datetime.now().strftime('%Y%m%d%H%M%S')}"),
        statistics=[
            {"id": "accuracy"},
            {"id": "f1score"},
            {"id": "misclassification"},
            {"id": "precision"},
        ]
    )
