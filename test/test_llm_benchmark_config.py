import csv
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from benchmarking.llm import get_llm_dataset_registry, get_llm_judge_registry, get_llm_metric_registry
from benchmarking.llm.dataset_registry import BehaviorSplitSpec, LLMDatasetSpec
from models import LLMBenchmarkConfig

TARGET = {
    "id": "llama3.1:8b", "name": "llama3.1:8b", "task": "language",
    "domain": "text", "model_type": "Ollama", "input_dimensionality": [1],
}
CATEGORIES = ("Fraud/Deception", "Malware/Hacking", "Privacy")


def make_config(**overrides) -> dict:
    config = {
        "target": {"model": TARGET},
        "attacks": [{
            "attack": {"id": "pair", "name": "PAIR", "task": "language", "parameters": [], "objective": "jailbreak"},
        }],
        "query_budget": 20,
        "benchmark_judges": [{"id": "jailjudge"}, {"id": "llama_guard"}],
        "metrics": [{"id": "asr"}, {"id": "refusal_benign"}, {"id": "judge_agreement"}],
    }
    config.update(overrides)
    return config


def write_split(path: Path, kind: str, per_category: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Index", "Goal", "Target", "Behavior", "Category", "Source"])
        index = 0
        for category in CATEGORIES:
            for _ in range(per_category):
                writer.writerow([index, f"{kind} goal {index}", f"Sure, {index}", "b", category, "Original"])
                index += 1


@pytest.fixture
def toy_dataset(tmp_path: Path) -> LLMDatasetSpec:
    write_split(tmp_path / "harmful.csv", "harmful", per_category=4)
    write_split(tmp_path / "benign.csv", "benign", per_category=4)
    return LLMDatasetSpec(
        dataset_id="toy",
        name="Toy",
        hf_repo="org/toy",
        revision="0" * 40,
        license="MIT",
        gated=False,
        splits={
            "harmful": BehaviorSplitSpec("harmful", "harmful.csv", sha256="unused", num_behaviors=12),
            "benign": BehaviorSplitSpec("benign", "benign.csv", sha256="unused", num_behaviors=12),
        },
        categories=CATEGORIES,
    )


# --- Configuration --- #
def test_valid_config_defaults() -> None:
    config = LLMBenchmarkConfig.model_validate(make_config())

    assert config.attacker == config.target.model
    assert config.attack_judge.id == "jailjudge"
    assert config.dataset.id == "jbb_behaviors"
    assert config.metric_parameters("asr") == {"ci_level": 0.95}


def test_metric_parameters_override_defaults() -> None:
    config = LLMBenchmarkConfig.model_validate(make_config(
        metrics=[{"id": "asr_at_budget", "parameters": {"k": 5}}],
    ))

    assert config.metric_parameters("asr_at_budget") == {"k": 5, "ci_level": 0.95}


@pytest.mark.parametrize("overrides, message", [
    ({"dataset": {"id": "unknown"}}, "Unknown LLM dataset"),
    ({"dataset": {"categories": ["Cooking"]}}, "Unknown categories"),
    ({"dataset": {"subset": 101}}, "exceeds the split size"),
    ({"benchmark_judges": [{"id": "unknown"}]}, "Unknown LLM judge"),
    ({"attack_judge": {"id": "llama_guard"}}, "cannot be used as attack judge"),
    ({"benchmark_judges": [{"id": "jailjudge"}, {"id": "jailjudge"}]}, "selected more than once"),
    ({"metrics": [{"id": "unknown"}]}, "Unknown LLM metric"),
    ({"query_budget": None}, "query_budget"),
    ({"query_budget": 0}, "greater than 0"),
    ({"metrics": [{"id": "asr", "parameters": {"alpha": 1}}]}, "Unknown parameters"),
    ({"dataset": {"include_benign": False}}, "needs the benign behaviors"),
    ({"benchmark_judges": [{"id": "jailjudge"}]}, "at least 2 benchmark judges"),
    ({"attacks": [], "metrics": [{"id": "resilience_gap"}]}, "needs at least one attack"),
    ({"attacks": [{"attack": {"id": "fgsm", "name": "FGSM", "task": "classification",
                              "parameters": [], "objective": "evasion"}}]}, "only supports 'jailbreak'"),
    ({"attacks": [{"attack": {"id": "gcg", "name": "GCG", "task": "language", "parameters": [],
                              "objective": "jailbreak"}}]}, "not supported by the LLM benchmark"),
])
def test_invalid_config_is_rejected(overrides: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        LLMBenchmarkConfig.model_validate(make_config(**overrides))


def test_protocol_warnings() -> None:
    config = LLMBenchmarkConfig.model_validate(make_config(
        attacks=[{"attack": {"id": "pair", "name": "PAIR", "task": "language", "parameters": []}, "query_budget": 5}],
        benchmark_judges=[{"id": "jailjudge"}],
        metrics=[{"id": "asr"}],
    ))

    warnings = config.protocol_warnings()
    assert any("may be inflated" in warning for warning in warnings)
    assert any("not goal-conditioned" in warning for warning in warnings)
    assert any("overrides the run budget (5 vs 20)" in warning for warning in warnings)
    assert config.budget_for("pair") == 5


def test_no_warnings_for_a_complete_protocol() -> None:
    config = LLMBenchmarkConfig.model_validate(make_config(benchmark_judges=[{"id": "llama_guard"}],
                                                           metrics=[{"id": "asr"}]))

    assert config.protocol_warnings() == []
    assert config.budget_for("pair") == 20


# --- Registries --- #
def test_registry_objects_are_frontend_ready() -> None:
    specs = [
        *get_llm_dataset_registry().list_specs(),
        *get_llm_judge_registry().list_specs(),
        *get_llm_metric_registry().list_specs(),
    ]
    for spec in specs:
        registered = spec.to_registered_object()
        assert registered.task == "language"
        assert registered.objective == "jailbreak"
        assert spec.info()["id"] == registered.id


def test_judge_types_are_loadable() -> None:
    from utils.load_model import JUDGE_TYPES

    for spec in get_llm_judge_registry().list_specs():
        assert spec.judge_type in get_args(JUDGE_TYPES)


def test_judge_threshold() -> None:
    jailjudge = get_llm_judge_registry().get("jailjudge")

    assert not jailjudge.is_jailbroken(2.0)
    assert jailjudge.is_jailbroken(3.0)
    assert not jailjudge.is_jailbroken(3.0, threshold=5.0)
    assert [spec.judge_id for spec in get_llm_judge_registry().list_specs(role="attack")] == ["jailjudge"]


# --- Dataset loading --- #
def test_load_behaviors(toy_dataset: LLMDatasetSpec, tmp_path: Path) -> None:
    behaviors = toy_dataset.load_behaviors(root=tmp_path, verify_hash=False)

    assert len(behaviors) == 24
    assert behaviors[0].behavior_id == "toy/harmful/0"
    assert behaviors[0].goal == "harmful goal 0"
    assert {behavior.kind for behavior in behaviors} == {"harmful", "benign"}


def test_subset_is_balanced_and_reproducible(toy_dataset: LLMDatasetSpec, tmp_path: Path) -> None:
    load = lambda seed: toy_dataset.load_behaviors(  # noqa: E731
        kinds=("harmful",), subset=6, seed=seed, root=tmp_path, verify_hash=False)

    behaviors = load(0)
    assert len(behaviors) == 6
    assert sorted(behavior.category for behavior in behaviors) == sorted(CATEGORIES * 2)
    assert behaviors == load(0)
    assert [int(b.behavior_id.rsplit("/", 1)[1]) for b in behaviors] == sorted(
        int(b.behavior_id.rsplit("/", 1)[1]) for b in behaviors)


def test_category_filter(toy_dataset: LLMDatasetSpec, tmp_path: Path) -> None:
    behaviors = toy_dataset.load_behaviors(kinds=("benign",), categories=["Privacy"], root=tmp_path,
                                           verify_hash=False)

    assert len(behaviors) == 4
    assert {behavior.category for behavior in behaviors} == {"Privacy"}


def test_hash_mismatch_and_missing_file(toy_dataset: LLMDatasetSpec, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sha256 mismatch"):
        toy_dataset.load_behaviors(root=tmp_path)
    with pytest.raises(FileNotFoundError, match="hf download org/toy"):
        toy_dataset.load_behaviors(root=tmp_path / "missing", verify_hash=False)


JBB = get_llm_dataset_registry().get("jbb_behaviors")


@pytest.mark.skipif(not (JBB.default_root() / "data").is_dir(), reason="JBB-Behaviors not downloaded")
def test_local_jbb_matches_registry() -> None:
    behaviors = JBB.load_behaviors()

    assert len(behaviors) == 200
    assert {behavior.category for behavior in behaviors} == set(JBB.categories)
