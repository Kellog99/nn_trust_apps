import json
from datetime import datetime
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from benchmarking import BenchmarkExecutor
from benchmarking.utils.evaluation import evaluate_attack
from models.reports import ReportAttackProps, AttackMetricsProps
from nn_trust import StatisticComposer, ModelAdapter
from test.test_single_attack import available_devices
from test.utils import get_dummy_cv_model, get_dummy_dataloader


@pytest.fixture
def model() -> ModelAdapter:
    return get_dummy_cv_model()


@pytest.fixture
def dataloader() -> DataLoader:
    return get_dummy_dataloader(num_samples=10)


@pytest.fixture
def attacks() -> list[dict]:
    return [
        {"id": "identitybaseline"},
        {"id": "contrastbaseline"},
        {"id": "gaussianbaseline"},
    ]


@pytest.fixture
def statistics() -> dict[str, dict]:
    return {
        "accuracy": {},
        "f1score": {},
        "misclassification": {},
        "precision": {},
    }


def test_statistic_composer_accepts_api_metric_metadata():
    """Display metadata returned by ``/info/metrics`` is not constructor input."""
    composer = StatisticComposer(statistics={
        "accuracy": {"device": torch.device("cpu")}
    }
    )

    assert set(composer._performance_stats) == {"accuracy"}


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
    tmp_path: Path = tmp_path / str(mse)
    tmp_path.mkdir(exist_ok=True, parents=True)
    checkpoint_path = tmp_path / "identitybaseline" / "log.pth"
    job_results_path = tmp_path / "identitybaseline" / "job_results.json"

    model.to(device)
    statistics = StatisticComposer()
    result = evaluate_attack(
        dataloader=dataloader,
        model=model,
        attack_id="identitybaseline",
        statistics=statistics,
        device=device,
        output_path=tmp_path,
        max_saved_elements=max_saved_elements,
    )

    assert result.id == "identitybaseline", "no identity"

    assert job_results_path.exists(), f"the file does not exist in {job_results_path}"
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
def test_execution(
        model: ModelAdapter,
        dataloader: DataLoader,
        attacks: list[dict],
        device: torch.device,
        tmp_path: Path,
        statistics: dict[str, dict],
):
    """
    Given N attacks that all succeed, execute_jobs must return exactly N
    ReportAttackProps, keyed by attack id, each with usable metrics.
    """
    executor = BenchmarkExecutor(
        device=device,
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
        for statistic_id in statistics:
            assert metric[statistic_id] is not None, f"Stat {statistic_id} is None"


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
        tmp_path=Path(f"./tmp/{datetime.now().strftime('%Y%m%d%H%M%S')}"),
        statistics={
            "accuracy": {},
            "f1score": {},
            "misclassification": {},
            "precision": {},
        }
    )
