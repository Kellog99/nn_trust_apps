import csv
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
import torch

from benchmarking.llm.budget import CallCounter, QueryBudgetExceeded, attach_query_counter, count_prompts
from benchmarking.llm.dataset_registry import JBB_CATEGORIES, get_llm_dataset_registry
from benchmarking.llm.runner import BASELINE_ID, LLMBenchmarkRunner, resolve_model_info
from models import JobResult, LLMBenchmarkConfig, ModelInfo
from nn_trust.attack.nlp.adapter import NLPModelAdapter
from nn_trust.attack.nlp.judge import BaseJudge, JudgeResult
from nn_trust.attack.nlp.types import GenerateOutput
from nn_trust.core import Task

REFUSAL, COMPLIANCE = "I'm sorry, I can't help with that.", "Sure, here is how to do it."


class FakeTarget(NLPModelAdapter):
    """Refuses a behavior goal sent as is, complies with any rewritten prompt."""

    def __init__(self, goals: set[str], fail_on: str | None = None):
        super().__init__(model=None, task=Task.Language)
        self.goals, self.fail_on, self.prompts = goals, fail_on, 0

    def query(self, messages, **kwargs) -> GenerateOutput:
        batch = messages if count_prompts(messages) > 1 or (
                isinstance(messages, list) and messages and not isinstance(messages[0], dict)) else [messages]
        texts = []
        for item in batch:
            text = item if isinstance(item, str) else item[-1]["content"]
            if text == self.fail_on:
                raise RuntimeError("target crashed")
            texts.append(REFUSAL if text in self.goals else COMPLIANCE)
        self.prompts += len(batch)
        return GenerateOutput(text=texts, sequences=torch.empty(0))


class KeywordJudge(BaseJudge):
    def __init__(self):
        self.calls = 0

    def evaluate(self, goal: str, response: str, **kwargs) -> JudgeResult:
        self.calls += 1
        return JudgeResult(score=10.0 if response.startswith("Sure") else 1.0)


class ConstantJudge(BaseJudge):
    def evaluate(self, goal: str, response: str, **kwargs) -> JudgeResult:
        return JudgeResult(score=1.0)


class FakeLoader:
    """The first judge loaded is the attack judge (phase A), then the benchmark judges (phase B)."""

    def __init__(self, goals: set[str], attack_judge: BaseJudge | None = None, fail_on: str | None = None):
        self.goals, self.fail_on = goals, fail_on
        self.attack_judge = attack_judge or KeywordJudge()
        self.adapters: list[FakeTarget] = []
        self.judges: list[tuple[str, BaseJudge]] = []
        self.releases = 0

    def adapter(self, info, *, max_new_tokens, n_ctx=None):
        adapter = FakeTarget(self.goals, self.fail_on if not self.adapters else None)
        self.adapters.append(adapter)
        return adapter

    def judge(self, info, spec):
        judge = self.attack_judge if not self.judges else KeywordJudge()
        self.judges.append((spec.judge_id, judge))
        return judge

    def release(self):
        self.releases += 1


# --- Fixtures --- #
@pytest.fixture
def model_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "model_repository"
    for folder, model_id in (("jailjudge-guard-q4", "jailjudge-guard-q4_k_m"),
                             ("llama-guard-3-8b-q4ks", "llama-guard-3-8b-q4ks")):
        (repo / folder).mkdir(parents=True)
        (repo / folder / "info.json").write_text(json.dumps({
            "id": model_id, "name": model_id, "task": "language", "model_type": "Llamacpp",
            "input_dimensionality": [1], "is_judge": True,
        }))
    return repo


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "jbb"
    for kind in ("harmful", "benign"):
        path = root / "data" / f"{kind}-behaviors.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Index", "Goal", "Target", "Behavior", "Category", "Source"])
            for index, category in enumerate(JBB_CATEGORIES):
                writer.writerow([index, f"{kind} goal {index}", "Sure", "b", category, "Original"])
    return root


def goals() -> set[str]:
    return {f"{kind} goal {index}" for kind in ("harmful", "benign") for index in range(len(JBB_CATEGORIES))}


def attack(attack_id: str, **parameters) -> dict:
    return {"attack": {"id": attack_id, "name": attack_id, "task": "language", "objective": "jailbreak",
                       "parameters": [{"id": key, "name": key, "default": value} for key, value in parameters.items()]}}


def make_config(tmp_path: Path, **overrides) -> LLMBenchmarkConfig:
    config = {
        "target": {"model": {"id": "fake-target", "name": "Fake", "task": "language", "model_type": "Ollama",
                             "input_dimensionality": [1], "repository": str(tmp_path)}},
        "dataset": {"subset": 4},
        "attacks": [attack("deepinception", max_iters=1)],
        "query_budget": 6,
        "benchmark_judges": [{"id": "jailjudge"}, {"id": "llama_guard"}],
        "metrics": [{"id": "asr"}, {"id": "asr_by_category"}, {"id": "resilience_gap"}, {"id": "judge_agreement"}],
        "options": {"output_path": str(tmp_path / "reports"), "gpu": False, "verbose": False},
    }
    config.update(overrides)
    return LLMBenchmarkConfig.model_validate(config)


def make_runner(config, model_repo, dataset_root, loader, benchmark_id="run") -> LLMBenchmarkRunner:
    runner = LLMBenchmarkRunner(config, benchmark_id=benchmark_id, model_repository=model_repo,
                                loader=loader, dataset_root=dataset_root)
    # The toy files replace JBB: pin their hashes as the registry does for the real ones.
    spec = get_llm_dataset_registry().get("jbb_behaviors")
    splits = {
        kind: dataclasses.replace(split, sha256=hashlib.sha256((dataset_root / split.filename).read_bytes()).hexdigest())
        for kind, split in spec.splits.items()
    }
    runner.dataset = dataclasses.replace(spec, splits=splits)
    return runner


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- Counter --- #
def test_counter_is_hard_and_sticky() -> None:
    counter = CallCounter()
    target = attach_query_counter(FakeTarget(set()), counter)
    counter.reset(budget=3)

    target.query(["a", "b"])
    with pytest.raises(QueryBudgetExceeded):
        target.query(["c", "d"])  # 2 + 2 > 3: the whole batch is refused
    with pytest.raises(QueryBudgetExceeded):
        target.query("e")  # still refused, although it would fit
    assert counter.used == 2 and counter.exhausted
    assert target.prompts == 2


@pytest.mark.parametrize("messages, n", [
    ("hello", 1),
    ([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}], 1),
    (["a", "b", "c"], 3),
    ([[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]], 2),
])
def test_count_prompts(messages, n: int) -> None:
    assert count_prompts(messages) == n


# --- End to end --- #
def test_run_end_to_end(tmp_path: Path, model_repo: Path, dataset_root: Path) -> None:
    loader = FakeLoader(goals())
    runner = make_runner(make_config(tmp_path), model_repo, dataset_root, loader)

    report = runner.run()

    folder = runner.folder
    assert folder == (tmp_path / "reports" / "run" / "fake-target" / "jbb_behaviors").resolve()
    baseline = read_jsonl(folder / BASELINE_ID / "records.jsonl")
    attacked = read_jsonl(folder / "deepinception" / "records.jsonl")
    assert [row["kind"] for row in baseline].count("benign") == 4  # generated, not measured
    assert len(attacked) == 4 and all(row["target_queries"] <= 6 for row in attacked)

    # Baseline refused, attack complied: ASR 0 -> 1 and gap 1, for both judges.
    for judge_id in ("jailjudge", "llama_guard"):
        assert report["jobs"][BASELINE_ID]["seeds"]["baseline"]["judges"][judge_id]["asr"]["value"] == 0.0
        seed = report["jobs"]["deepinception"]["seeds"]["0"]["judges"][judge_id]
        assert seed["asr"]["value"] == 1.0
        assert seed["resilience_gap"]["value"] == 1.0
        assert sum(category["n"] for category in seed["asr_by_category"].values()) == 4
    # Both judges always say "jailbroken": kappa is undefined and saved as null.
    assert report["jobs"]["deepinception"]["seeds"]["0"]["judge_agreement"] == {"jailjudge|llama_guard": None}
    assert "refusal_benign" not in json.dumps(report["jobs"])

    # Benchmark judges ran after generation, one at a time, each on every record.
    assert [judge_id for judge_id, _ in loader.judges] == ["jailjudge", "jailjudge", "llama_guard"]
    assert loader.releases == 3
    assert report["cost"]["benchmark_judge_calls"] == 2 * (8 + 4)
    # Attacker is a distinct adapter: its calls are never target queries.
    target, attacker = loader.adapters
    assert target is not attacker
    assert report["cost"]["target_queries"] == target.prompts

    for job_id in (BASELINE_ID, "deepinception"):
        job = JobResult.model_validate_json((folder / job_id / "job_results.json").read_text())
        assert job.status == "finished" and job.progress == job.total

    manifest = json.loads((folder / "manifest.json").read_text())
    assert len(manifest["dataset"]["behavior_ids"]) == 8
    assert manifest["config"]["query_budget"] == 6
    assert any("over-refusal is not computed" in note for note in manifest["notes"])


def test_budget_stops_attack_and_keeps_state(tmp_path: Path, model_repo: Path, dataset_root: Path) -> None:
    # The attack judge never declares success, so AutoDAN would go on: the
    # second population (4 more prompts) does not fit in the budget of 6.
    config = make_config(tmp_path, attacks=[attack("autodan", population_size=4, max_iters=5)],
                         dataset={"subset": 2, "include_benign": False})
    runner = make_runner(config, model_repo, dataset_root, FakeLoader(goals(), attack_judge=ConstantJudge()))

    runner.run()

    rows = read_jsonl(runner.folder / "autodan" / "records.jsonl")
    assert len(rows) == 2
    for row in rows:
        assert row["budget_exhausted"] is True
        assert row["target_queries"] == 4
        assert row["best_response"] == COMPLIANCE  # state of the last completed step
        assert row["error"] is None


def test_resume_runs_only_missing_units(tmp_path: Path, model_repo: Path, dataset_root: Path) -> None:
    config = make_config(tmp_path)
    make_runner(config, model_repo, dataset_root, FakeLoader(goals())).run()
    folder = make_runner(config, model_repo, dataset_root, FakeLoader(goals())).folder
    records = folder / "deepinception" / "records.jsonl"
    lines = records.read_text().splitlines(keepends=True)
    records.write_text("".join(lines[:-1]))

    loader = FakeLoader(goals())
    report = make_runner(config, model_repo, dataset_root, loader).run()

    target = loader.adapters[0]
    assert target.prompts == 1  # only the removed unit was generated again
    assert len(read_jsonl(records)) == 4
    assert report["jobs"]["deepinception"]["seeds"]["0"]["judges"]["jailjudge"]["asr"]["n"] == 4


def test_resume_refuses_a_different_config(tmp_path: Path, model_repo: Path, dataset_root: Path) -> None:
    make_runner(make_config(tmp_path), model_repo, dataset_root, FakeLoader(goals())).run()

    changed = make_config(tmp_path, query_budget=7)
    with pytest.raises(ValueError, match="use a new benchmark id"):
        make_runner(changed, model_repo, dataset_root, FakeLoader(goals())).run()


def test_errors_are_recorded_and_excluded(tmp_path: Path, model_repo: Path, dataset_root: Path) -> None:
    runner = make_runner(make_config(tmp_path), model_repo, dataset_root,
                         FakeLoader(goals(), fail_on="harmful goal 0"))

    report = runner.run()

    assert report["jobs"][BASELINE_ID]["n_errors"] == 1
    baseline = report["jobs"][BASELINE_ID]["seeds"]["baseline"]["judges"]["jailjudge"]
    assert baseline["n"] == report["jobs"]["deepinception"]["seeds"]["0"]["judges"]["jailjudge"]["n"] - 1


def test_resolve_model_info(model_repo: Path) -> None:
    info = resolve_model_info("jailjudge-guard-q4_k_m", model_repo)
    assert info.repository == str(model_repo / "jailjudge-guard-q4")

    given = ModelInfo(id="x", name="x", task="language", input_dimensionality=[1], repository="/somewhere")
    assert resolve_model_info(given, model_repo) is given
    with pytest.raises(FileNotFoundError):
        resolve_model_info("missing", model_repo)
