import csv
import json
import math
from pathlib import Path

import pytest

from benchmarking.llm.metrics import (
    asr,
    asr_by_category,
    cohen_kappa,
    judge_agreement,
    refusal_rate,
    resilience_gap,
    wilson_interval,
)

JBB_DIR = Path(__file__).resolve().parents[1] / "data" / "benchmarks" / "jbb_behaviors"


# --- Wilson --- #
@pytest.mark.parametrize("successes, n, low, high", [
    # Reference values: Newcombe (1998), Statistics in Medicine 17, Table I, method 3.
    (81, 263, 0.2553, 0.3662),
    (15, 148, 0.0624, 0.1605),
    (0, 20, 0.0000, 0.1611),
    (1, 29, 0.0061, 0.1718),
])
def test_wilson_matches_published_values(successes: int, n: int, low: float, high: float) -> None:
    assert wilson_interval(successes, n) == pytest.approx((low, high), abs=1e-4)


def test_wilson_stays_in_unit_interval_and_narrows_with_n() -> None:
    assert wilson_interval(0, 5)[0] == 0.0
    assert wilson_interval(5, 5)[1] == 1.0
    small, large = wilson_interval(30, 100), wilson_interval(300, 1000)
    assert large[1] - large[0] < small[1] - small[0]
    assert wilson_interval(30, 100, ci_level=0.99)[0] < small[0]


@pytest.mark.parametrize("successes, n, ci_level", [(0, 0, 0.95), (3, 2, 0.95), (1, 2, 1.0)])
def test_wilson_rejects_invalid_input(successes: int, n: int, ci_level: float) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, n, ci_level)


# --- Rates --- #
def test_asr() -> None:
    result = asr([True] * 30 + [False] * 70)

    assert result.value == 0.3
    assert (result.successes, result.n) == (30, 100)
    assert result.low < 0.3 < result.high


def test_asr_by_category() -> None:
    result = asr_by_category([True, False, True, True], ["Privacy", "Privacy", "Fraud", "Fraud"])

    assert list(result) == ["Fraud", "Privacy"]
    assert result["Fraud"].value == 1.0
    assert result["Privacy"].value == 0.5
    with pytest.raises(ValueError):
        asr_by_category([True], ["Privacy", "Fraud"])


def test_refusal_rate() -> None:
    assert refusal_rate([True, False, False, False]).value == 0.25


def test_resilience_gap() -> None:
    result = resilience_gap(baseline_jailbroken=[True, False, False, False],
                            attack_jailbroken=[True, True, True, False])

    assert result.value == pytest.approx(0.5)
    assert (result.baseline.value, result.attack.value) == (0.25, 0.75)
    with pytest.raises(ValueError):
        resilience_gap([True], [True, False])


# --- Agreement --- #
def test_cohen_kappa() -> None:
    a = [True, True, False, False]
    assert cohen_kappa(a, a) == 1.0
    assert cohen_kappa(a, [False, False, True, True]) == -1.0
    # po = 0.75, pe = 0.5 -> kappa = 0.5
    assert cohen_kappa(a, [True, False, False, False]) == pytest.approx(0.5)
    assert math.isnan(cohen_kappa([True, True], [True, True]))


def test_judge_agreement_is_pairwise() -> None:
    result = judge_agreement({"b": [True, False], "a": [True, False], "c": [False, True]})

    assert list(result) == [("a", "b"), ("a", "c"), ("b", "c")]
    assert result[("a", "b")] == 1.0
    with pytest.raises(ValueError):
        judge_agreement({"a": [True]})


# --- Regression on Phase 1 results (roadmap §5.5) --- #
@pytest.mark.skipif(not (JBB_DIR / "data" / "judge-comparison.csv").is_file(), reason="JBB-Behaviors not downloaded")
def test_kappa_reproduces_reference_judge() -> None:
    with open(JBB_DIR / "data" / "judge-comparison.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    human = [row["human_majority"] == "1" for row in rows]
    llama3 = [row["llama3_cf"] == "1" for row in rows]

    assert judge_agreement({"human": human, "llama3_cf": llama3})[("human", "llama3_cf")] == pytest.approx(0.805, abs=1e-3)


ROWS = JBB_DIR / "results" / "jailjudge_goal" / "rows.jsonl"


@pytest.mark.skipif(not ROWS.is_file(), reason="Phase 1 JailJudge results not available")
def test_kappa_reproduces_phase1_jailjudge() -> None:
    with open(ROWS, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    human = [row["human_majority"] == 1 for row in rows]
    jailjudge = [row["score"] > 2 for row in rows]

    assert len(rows) == 300
    assert cohen_kappa(human, jailjudge) == pytest.approx(0.728, abs=1e-3)
