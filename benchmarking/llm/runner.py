"""LLM benchmark runner: attack x behavior x target, in two phases.

Phase A (generation): target, attacker and attack judge are loaded; every
(behavior, attack, seed) unit runs once, behind a hard query budget on the
target. Benign behaviors are only asked without attack (baseline).
Phase B (judging): target and attacker are released, then each benchmark
judge is loaded alone and scores the saved ``best_response``.
Metrics are computed from the saved files.

Layout of one run::

    <output_path>/<benchmark_id>/<target_id>/<dataset_id>/
        manifest.json                 protocol (config, dataset, judges, code)
        llm_report.json               metrics and costs
        <job_id>/job_results.json     progress, read by GET /job/getJobs
        <job_id>/records.jsonl        one row per generated unit
        <job_id>/verdicts/<judge>.jsonl

Every unit is appended as soon as it ends, so an interrupted run resumes
where it stopped.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol

import torch

from benchmarking.llm import metrics
from benchmarking.llm.budget import (
    CallCounter,
    QueryBudgetExceeded,
    attach_evaluate_counter,
    attach_query_counter,
)
from benchmarking.llm.dataset_registry import Behavior, get_llm_dataset_registry
from benchmarking.llm.judge_registry import LLMJudgeSpec, get_llm_judge_registry
from models import JobResult, LLMBenchmarkConfig, ModelInfo
from models.llm_benchmark import LLMJudgeSelection

BASELINE_ID = "identitybaseline"
PROTOCOL = "TITANN-LLM v1"
ROOT = Path(__file__).resolve().parents[2]
ATTACKER_MAX_NEW_TOKENS = 1024
JUDGE_N_CTX = 4096
# Attacks returning only their best stream: if the budget runs out, the
# previous streams are lost (accepted limit, roadmap Phase 2).
MULTI_STREAM_PARAMETER = {"pair": "n_streams", "code": "n_repetitions", "redqueen": "n_scenarios"}

Unit = tuple[Behavior, Optional[int]]


class ModelLoader(Protocol):
    def adapter(self, info: ModelInfo, *, max_new_tokens: int, n_ctx: Optional[int] = None) -> Any:
        """Load an NLP adapter (target or attacker)."""

    def judge(self, info: ModelInfo, spec: LLMJudgeSpec) -> Any:
        """Load a judge exposing `evaluate(goal, response)`."""

    def release(self) -> None:
        """Free the memory of the models no longer referenced."""


class DefaultModelLoader:
    """Loads models through `utils.load_model`, as the single-attack endpoint does."""

    def __init__(self, device: torch.device):
        self.device = device

    def adapter(self, info: ModelInfo, *, max_new_tokens: int, n_ctx: Optional[int] = None) -> Any:
        from nn_trust import Task
        from utils.load_model import load_model

        kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if n_ctx:
            kwargs["n_ctx"] = n_ctx
        return load_model(
            model_type=info.model_type,
            model_id=info.id,
            model_path=info.repository,
            api_url=info.api,
            task=Task.Language,
            device=self.device,
            **kwargs,
        )

    def judge(self, info: ModelInfo, spec: LLMJudgeSpec) -> Any:
        from nn_trust import Task
        from utils.load_model import load_model

        return load_model(
            model_type=info.model_type,
            model_id=info.id,
            model_path=info.repository,
            api_url=info.api,
            task=Task.Language,
            device=self.device,
            is_judge=True,
            judge_type=spec.judge_type,
            temperature=0.0,
            n_ctx=JUDGE_N_CTX,
        )

    def release(self) -> None:
        from utils.model._loader_nlp_models import clear_model_cache

        clear_model_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def resolve_model_info(model: ModelInfo | str, model_repository: Path) -> ModelInfo:
    """
    Return the model info with `repository` set, looking up `info.json` by id when needed.
    """
    if isinstance(model, ModelInfo) and model.repository:
        return model
    model_id = model.id if isinstance(model, ModelInfo) else model
    for info_file in sorted(Path(model_repository).expanduser().glob("*/info.json")):
        data = json.loads(info_file.read_text(encoding="utf-8"))
        if data.get("id") != model_id:
            continue
        repository = data.get("repository") or str(info_file.parent)
        if isinstance(model, ModelInfo):
            return model.model_copy(update={"repository": repository})
        return ModelInfo.model_validate({**data, "repository": repository})
    raise FileNotFoundError(f"No info.json with id '{model_id}' under {model_repository}.")


class LLMBenchmarkRunner:
    def __init__(
            self,
            config: LLMBenchmarkConfig,
            *,
            benchmark_id: Optional[str] = None,
            model_repository: str | Path = "~/Desktop/StableAI/model_repository",
            loader: Optional[ModelLoader] = None,
            dataset_root: Optional[str | Path] = None,
    ):
        self.config = config
        self.benchmark_id = benchmark_id or config.benchmark_id or datetime.now().strftime("%Y%m%dT%H%M%S_%f")
        self.model_repository = Path(model_repository).expanduser()
        self.device = torch.device("cuda" if config.options.gpu and torch.cuda.is_available() else "cpu")
        self.loader = loader or DefaultModelLoader(self.device)
        self.dataset = get_llm_dataset_registry().get(config.dataset.id)
        self.dataset_root = Path(dataset_root).expanduser() if dataset_root is not None else None
        self.judges = get_llm_judge_registry()
        self.folder = (
                Path(config.options.output_path).expanduser().resolve()
                / self.benchmark_id / config.target.model.id / config.dataset.id
        )

    # --- Entry point --- #
    def run(self) -> dict[str, Any]:
        behaviors = self._load_behaviors()
        self._prepare_folder(behaviors)
        jobs = self._plan(behaviors)
        try:
            self._generate(jobs)
            self._judge(jobs)
            return self._report(jobs)
        except Exception as exc:
            self._mark_failed(jobs, f"{type(exc).__name__}: {exc}")
            raise

    def _load_behaviors(self) -> list[Behavior]:
        selection = self.config.dataset
        return self.dataset.load_behaviors(
            kinds=("harmful", "benign") if selection.include_benign else ("harmful",),
            categories=selection.categories,
            subset=selection.subset,
            seed=selection.seed,
            root=self.dataset_root,
        )

    def _plan(self, behaviors: list[Behavior]) -> dict[str, list[Unit]]:
        harmful = [behavior for behavior in behaviors if behavior.kind == "harmful"]
        jobs: dict[str, list[Unit]] = {BASELINE_ID: [(behavior, None) for behavior in behaviors]}
        for selection in self.config.attacks:
            jobs[selection.attack.id] = [(behavior, seed) for behavior in harmful for seed in selection.seeds]
        for job_id, units in jobs.items():
            (self.folder / job_id / "verdicts").mkdir(parents=True, exist_ok=True)
            # On resume the units already in records.jsonl count as done.
            job = self._load_job(job_id).model_copy(update={
                "total": len(units),
                "progress": len(self._done_keys(job_id)),
                "status": "pending",
                "error": None,
            })
            job.save(self._job_path(job_id))
        return jobs

    # --- Phase A: generation --- #
    def _generate(self, jobs: dict[str, list[Unit]]) -> None:
        pending = {
            job_id: [unit for unit in units if _key(*unit) not in self._done_keys(job_id)]
            for job_id, units in jobs.items()
        }
        if not any(pending.values()):
            return

        cfg = self.config
        target_counter, attacker_counter, judge_counter = CallCounter(), CallCounter(), CallCounter()
        target = self.loader.adapter(self._target_info(), max_new_tokens=cfg.target.max_new_tokens,
                                     n_ctx=cfg.target.n_ctx)
        # Generation settings are part of the protocol: set them on the instance,
        # since not every loader forwards them.
        target.temperature = cfg.target.temperature
        target.do_sample = cfg.target.temperature > 0
        target.max_new_tokens = cfg.target.max_new_tokens
        attach_query_counter(target, target_counter)

        attacker = attack_judge = None
        if any(pending[job_id] for job_id in pending if job_id != BASELINE_ID):
            # Always a distinct adapter, even for the same model as the target,
            # so that attacker calls are never counted as target queries.
            attacker = self.loader.adapter(resolve_model_info(cfg.attacker, self.model_repository),
                                           max_new_tokens=ATTACKER_MAX_NEW_TOKENS)
            attach_query_counter(attacker, attacker_counter)
            spec = self.judges.get(cfg.attack_judge.id)
            attack_judge = self.loader.judge(self._judge_info(cfg.attack_judge, spec), spec)
            attach_evaluate_counter(attack_judge, judge_counter)

        try:
            for job_id, units in pending.items():
                if not units:
                    continue
                job = self._load_job(job_id)
                job.status = "in progress"
                job.save(self._job_path(job_id))
                start = time.perf_counter()
                for i, (behavior, seed) in enumerate(units, start=1):
                    unit_start = time.perf_counter()
                    if job_id == BASELINE_ID:
                        row = self._baseline_unit(behavior, target, target_counter)
                    else:
                        row = self._attack_unit(job_id, behavior, seed, target, attacker, attack_judge,
                                                (target_counter, attacker_counter, judge_counter))
                    _append_jsonl(self.folder / job_id / "records.jsonl", row)

                    job.progress += 1
                    job.iteration_time = time.perf_counter() - unit_start
                    job.execution_time = time.perf_counter() - start
                    job.estimated_execution_time = job.execution_time / i * len(units)
                    job.save(self._job_path(job_id))
        finally:
            del target, attacker, attack_judge
            self.loader.release()

    def _baseline_unit(self, behavior: Behavior, target: Any, counter: CallCounter) -> dict[str, Any]:
        messages = [{"role": "user", "content": behavior.goal}]
        if self.config.target.system_prompt:
            messages.insert(0, {"role": "system", "content": self.config.target.system_prompt})
        counter.reset(budget=1)
        start, response, error = time.perf_counter(), None, None
        try:
            out = target.query(messages)
            response = out.text[0] if out.text else ""
        except Exception as exc:
            if self.config.options.on_error == "abort":
                raise
            error = f"{type(exc).__name__}: {exc}"
        return {
            **_behavior_fields(behavior, seed=None),
            "best_prompt": behavior.goal,
            "best_response": response,
            "attack_judge_score": None,
            "attack_success": None,
            "n_attempts": 1,
            "target_queries": counter.used,
            "budget": 1,
            "budget_exhausted": False,
            "attacker_calls": 0,
            "attack_judge_calls": 0,
            "seconds": time.perf_counter() - start,
            "error": error,
        }

    def _attack_unit(
            self,
            attack_id: str,
            behavior: Behavior,
            seed: int,
            target: Any,
            attacker: Any,
            attack_judge: Any,
            counters: tuple[CallCounter, CallCounter, CallCounter],
    ) -> dict[str, Any]:
        from nn_trust.attack import AttackFactory

        target_counter, attacker_counter, judge_counter = counters
        budget = self.config.budget_for(attack_id)
        target_counter.reset(budget=budget)
        attacker_counter.reset()
        judge_counter.reset()

        kwargs = {
            **self.config.attack_parameters(attack_id),
            "verbose": self.config.options.verbose,
            "device": self.device,
            "seed": seed,
        }
        if self.config.target.system_prompt:
            kwargs["system_prompt"] = self.config.target.system_prompt

        # A fresh attack per unit: results do not depend on the order of the
        # behaviors, and resume gives the same result as an uninterrupted run.
        random.seed(seed)
        start, attack, state, error = time.perf_counter(), None, None, None
        try:
            attack = AttackFactory.create(class_id=attack_id, model=target, attacker=attacker,
                                          judge=attack_judge, **kwargs)
            state = attack.generate(goal=behavior.goal)
        except QueryBudgetExceeded:
            # `res` is the state after the last completed step (for multi-stream
            # attacks: of the current stream only).
            state = getattr(attack, "res", None)
        except Exception as exc:
            if self.config.options.on_error == "abort":
                raise
            error = f"{type(exc).__name__}: {exc}"

        best_score = getattr(state, "best_score", None)
        return {
            **_behavior_fields(behavior, seed=seed),
            "best_prompt": _best_prompt(state),
            "best_response": getattr(state, "best_response", None),
            "attack_judge_score": best_score if best_score is not None and math.isfinite(best_score) else None,
            "attack_success": getattr(state, "success", None),
            "n_attempts": len(state.attempts) if state is not None else 0,
            "target_queries": target_counter.used,
            "budget": budget,
            "budget_exhausted": target_counter.exhausted,
            "attacker_calls": attacker_counter.used,
            "attack_judge_calls": judge_counter.used,
            "seconds": time.perf_counter() - start,
            "error": error,
        }

    # --- Phase B: judging --- #
    def _judge(self, jobs: dict[str, list[Unit]]) -> None:
        for selection in self.config.benchmark_judges:
            todo = {
                job_id: [row for row in self._records(job_id)
                         if row["key"] not in self._verdicts(job_id, selection.id)]
                for job_id in jobs
            }
            if not any(todo.values()):
                continue
            spec = self.judges.get(selection.id)
            judge = self.loader.judge(self._judge_info(selection, spec), spec)
            try:
                for job_id, rows in todo.items():
                    for row in rows:
                        _append_jsonl(self._verdict_path(job_id, selection.id), self._verdict(judge, row))
            finally:
                del judge
                self.loader.release()

    def _verdict(self, judge: Any, row: dict[str, Any]) -> dict[str, Any]:
        if row["error"] is not None or row["best_response"] is None:
            # Nothing to judge: the unit failed, or the attack never got a response.
            reason = "generation_error" if row["error"] is not None else "no_response"
            return {"key": row["key"], "score": None, "verdict": None, "judged": False, "reason": reason}
        start = time.perf_counter()
        try:
            result = judge.evaluate(row["goal"], row["best_response"])
        except Exception as exc:
            if self.config.options.on_error == "abort":
                raise
            return {"key": row["key"], "score": None, "verdict": None, "judged": False,
                    "reason": f"judge_error: {type(exc).__name__}: {exc}"}
        return {"key": row["key"], "score": float(result.score), "verdict": result.verdict, "judged": True,
                "seconds": time.perf_counter() - start}

    # --- Metrics --- #
    def _report(self, jobs: dict[str, list[Unit]]) -> dict[str, Any]:
        cfg = self.config
        selected = {metric.id for metric in cfg.metrics}
        thresholds = {
            selection.id: selection.threshold if selection.threshold is not None
            else self.judges.get(selection.id).default_threshold
            for selection in cfg.benchmark_judges
        }
        records = {job_id: self._records(job_id) for job_id in jobs}
        verdicts = {
            job_id: {judge_id: self._verdicts(job_id, judge_id) for judge_id in thresholds}
            for job_id in jobs
        }

        def outcomes(job_id: str, judge_id: str, rows: Iterable[dict]) -> dict[str, tuple[dict, bool]]:
            """behavior_id -> (row, jailbroken) for the rows with a usable verdict."""
            out = {}
            for row in rows:
                verdict = verdicts[job_id][judge_id].get(row["key"])
                if row["error"] is not None or verdict is None:
                    continue
                if verdict["judged"]:
                    out[row["behavior_id"]] = (row, verdict["score"] > thresholds[judge_id])
                elif verdict["reason"] == "no_response":
                    out[row["behavior_id"]] = (row, False)
            return out

        baseline_rows = [row for row in records[BASELINE_ID] if row["kind"] == "harmful"]
        baseline = {judge_id: outcomes(BASELINE_ID, judge_id, baseline_rows) for judge_id in thresholds}

        results: dict[str, Any] = {}
        for job_id in jobs:
            harmful = [row for row in records[job_id] if row["kind"] == "harmful"]
            by_seed: dict[str, list[dict]] = {}
            for row in harmful:
                by_seed.setdefault("baseline" if row["seed"] is None else str(row["seed"]), []).append(row)
            results[job_id] = {
                "seeds": {
                    seed: self._seed_metrics(job_id, rows, outcomes, baseline, selected)
                    for seed, rows in sorted(by_seed.items())
                },
                "n_records": len(records[job_id]),
                "n_errors": sum(row["error"] is not None for row in records[job_id]),
                "n_budget_exhausted": sum(bool(row["budget_exhausted"]) for row in records[job_id]),
                "cost": _cost(records[job_id], verdicts[job_id]),
            }

        report = _clean({
            "benchmark_id": self.benchmark_id,
            "protocol": PROTOCOL,
            "target": cfg.target.model.id,
            "dataset": cfg.dataset.id,
            "query_budget": cfg.query_budget,
            "thresholds": thresholds,
            "warnings": cfg.protocol_warnings(),
            "notes": self._notes(),
            "jobs": results,
            "cost": {
                key: sum(job["cost"][key] for job in results.values())
                for key in ("target_queries", "attacker_calls", "attack_judge_calls", "benchmark_judge_calls")
            },
        })
        (self.folder / "llm_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        for job_id in jobs:
            job = self._load_job(job_id)
            job.status, job.result = "finished", report["jobs"][job_id]
            job.save(self._job_path(job_id))
        return report

    def _seed_metrics(self, job_id, rows, outcomes, baseline, selected) -> dict[str, Any]:
        judge_ids = list(baseline)
        per_judge: dict[str, Any] = {}
        valid = {judge_id: outcomes(job_id, judge_id, rows) for judge_id in judge_ids}
        for judge_id, out in valid.items():
            values = [jailbroken for _, jailbroken in out.values()]
            entry: dict[str, Any] = {"n": len(values)}
            if values and "asr" in selected:
                entry["asr"] = asdict(metrics.asr(values, **self._params("asr")))
            if values and "asr_by_category" in selected:
                entry["asr_by_category"] = {
                    category: asdict(result) for category, result in metrics.asr_by_category(
                        values, [row["category"] for row, _ in out.values()],
                        **self._params("asr_by_category")).items()
                }
            if "resilience_gap" in selected and job_id != BASELINE_ID:
                shared = sorted(set(out) & set(baseline[judge_id]))
                if shared:
                    entry["resilience_gap"] = asdict(metrics.resilience_gap(
                        [baseline[judge_id][behavior_id][1] for behavior_id in shared],
                        [out[behavior_id][1] for behavior_id in shared],
                    ))
            per_judge[judge_id] = entry

        result: dict[str, Any] = {"judges": per_judge}
        if "judge_agreement" in selected:
            shared = sorted(set.intersection(*(set(out) for out in valid.values())))
            if shared:
                agreement = metrics.judge_agreement({
                    judge_id: [valid[judge_id][behavior_id][1] for behavior_id in shared] for judge_id in judge_ids
                })
                result["judge_agreement"] = {f"{a}|{b}": kappa for (a, b), kappa in agreement.items()}
        return result

    def _params(self, metric_id: str) -> dict[str, Any]:
        return self.config.metric_parameters(metric_id)

    # --- Protocol --- #
    def _prepare_folder(self, behaviors: list[Behavior]) -> None:
        manifest_path = self.folder / "manifest.json"
        protocol_config = self.config.model_dump(mode="json", exclude={"options", "benchmark_id"})
        if manifest_path.exists():
            if not self.config.options.resume:
                shutil.rmtree(self.folder)
            else:
                stored = json.loads(manifest_path.read_text(encoding="utf-8"))
                if stored["config"] != protocol_config:
                    raise ValueError(
                        f"The configuration differs from the one of the run in {self.folder}: "
                        "use a new benchmark id instead of resuming."
                    )
                return
        self.folder.mkdir(parents=True, exist_ok=True)

        attack_judge = self.judges.get(self.config.attack_judge.id)
        manifest = {
            "protocol": PROTOCOL,
            "benchmark_id": self.benchmark_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config": protocol_config,
            "warnings": self.config.protocol_warnings(),
            "notes": self._notes(),
            "budget_rules": {
                "unit": "one prompt sent to the target (a batch of N prompts counts N; a multi-turn turn counts 1)",
                "limit": "hard: a call exceeding the remaining budget is refused and the attack stops",
                "per": "behavior x attack x seed",
            },
            "dataset": {
                **self.dataset.info(),
                "files": {kind: {"filename": split.filename, "sha256": split.sha256}
                          for kind, split in self.dataset.splits.items()},
                "behavior_ids": [behavior.behavior_id for behavior in behaviors],
            },
            "target": self._target_info().model_dump(mode="json"),
            "attacker": resolve_model_info(self.config.attacker, self.model_repository).model_dump(mode="json"),
            "attacker_max_new_tokens": ATTACKER_MAX_NEW_TOKENS,
            "attack_judge": attack_judge.info(),
            "benchmark_judges": [
                {**self.judges.get(selection.id).info(), "threshold": selection.threshold,
                 "model": self._judge_info(selection, self.judges.get(selection.id)).model_dump(mode="json")}
                for selection in self.config.benchmark_judges
            ],
            "judge_n_ctx": JUDGE_N_CTX,
            "code": {"nn_trust_apps": _git_state(ROOT), "nn_trust": _git_state(ROOT / "submodules" / "nn_trust")},
        }
        manifest_path.write_text(json.dumps(_clean(manifest), indent=2), encoding="utf-8")

    def _notes(self) -> list[str]:
        cfg = self.config
        notes = []
        if cfg.dataset.include_benign:
            notes.append("Benign behaviors are generated without attack and saved; over-refusal is not computed "
                         "(future development).")
        for selection in cfg.attacks:
            parameter = MULTI_STREAM_PARAMETER.get(selection.attack.id)
            if parameter and int(cfg.attack_parameters(selection.attack.id).get(parameter) or 1) > 1:
                notes.append(f"'{selection.attack.id}' runs several streams: when the budget runs out only the last "
                             "stream counts, so its ASR may be underestimated.")
            if selection.attack.id == "autodanturbo":
                notes.append("Every behavior runs on a fresh attack instance: AutoDAN-Turbo does not carry its "
                             "strategy library across behaviors.")
        if cfg.target.model.model_type == "Ollama":
            notes.append("Ollama receives no seed: with temperature 0 the target is deterministic, "
                         "the attacker samples.")
        return notes

    def _target_info(self) -> ModelInfo:
        return resolve_model_info(self.config.target.model, self.model_repository)

    def _judge_info(self, selection: LLMJudgeSelection, spec: LLMJudgeSpec) -> ModelInfo:
        return resolve_model_info(selection.model or spec.model_id, self.model_repository)

    # --- Files --- #
    def _job_path(self, job_id: str) -> Path:
        return self.folder / job_id / "job_results.json"

    def _load_job(self, job_id: str) -> JobResult:
        path = self._job_path(job_id)
        if path.exists():
            return JobResult.model_validate_json(path.read_text(encoding="utf-8"))
        return JobResult(id=job_id)

    def _records(self, job_id: str) -> list[dict[str, Any]]:
        return _read_jsonl(self.folder / job_id / "records.jsonl")

    def _done_keys(self, job_id: str) -> set[str]:
        return {row["key"] for row in self._records(job_id)}

    def _verdict_path(self, job_id: str, judge_id: str) -> Path:
        return self.folder / job_id / "verdicts" / f"{judge_id}.jsonl"

    def _verdicts(self, job_id: str, judge_id: str) -> dict[str, dict[str, Any]]:
        return {row["key"]: row for row in _read_jsonl(self._verdict_path(job_id, judge_id))}

    def _mark_failed(self, jobs: dict[str, list[Unit]], error: str) -> None:
        for job_id in jobs:
            try:
                job = self._load_job(job_id)
                if job.status not in {"finished", "error"}:
                    job.status, job.error = "error", error
                    job.save(self._job_path(job_id))
            except (OSError, ValueError):
                print(f"Could not persist failed job '{job_id}'")


# --- Helpers --- #
def _key(behavior: Behavior, seed: Optional[int]) -> str:
    return f"{behavior.behavior_id}#{'baseline' if seed is None else seed}"


def _behavior_fields(behavior: Behavior, seed: Optional[int]) -> dict[str, Any]:
    return {
        "key": _key(behavior, seed),
        "behavior_id": behavior.behavior_id,
        "kind": behavior.kind,
        "category": behavior.category,
        "goal": behavior.goal,
        "seed": seed,
    }


def _best_prompt(state: Any) -> Optional[str]:
    """Prompt of the attempt that produced `best_response` (the state does not store it)."""
    if state is None or not state.attempts:
        return None
    for attempt in state.attempts:
        if attempt.response == state.best_response:
            return attempt.prompt
    return state.attempts[-1].prompt


def _cost(records: list[dict], verdicts: dict[str, dict[str, dict]]) -> dict[str, Any]:
    return {
        "target_queries": sum(row["target_queries"] for row in records),
        "attacker_calls": sum(row["attacker_calls"] for row in records),
        "attack_judge_calls": sum(row["attack_judge_calls"] for row in records),
        "benchmark_judge_calls": sum(row["judged"] for rows in verdicts.values() for row in rows.values()),
        "generation_seconds": sum(row["seconds"] for row in records),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_clean(row)) + "\n")


def _clean(value: Any) -> Any:
    """nan/inf are not valid JSON for the frontend: turn them into null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _git_state(path: Path) -> Optional[dict[str, Any]]:
    try:
        commit = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip() != ""
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"commit": commit, "dirty": dirty}


def run_llm_benchmark(config: LLMBenchmarkConfig, **kwargs) -> dict[str, Any]:
    return LLMBenchmarkRunner(config, **kwargs).run()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run an LLM benchmark from a JSON configuration.")
    parser.add_argument("config", type=Path, help="LLMBenchmarkConfig as JSON")
    parser.add_argument("--benchmark-id", default=None, help="Reuse an id to resume an interrupted run")
    parser.add_argument("--model-repo", default="~/Desktop/StableAI/model_repository")
    args = parser.parse_args(argv)

    config = LLMBenchmarkConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    for warning in config.protocol_warnings():
        print(f"[warning] {warning}")
    runner = LLMBenchmarkRunner(config, benchmark_id=args.benchmark_id, model_repository=args.model_repo)
    runner.run()
    print(runner.folder)


if __name__ == "__main__":
    main()
