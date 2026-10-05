"""App-owned registry of LLM behavior datasets (goal sets for jailbreak benchmarks).

Datasets are never committed: they are downloaded at runtime under
``data/benchmarks/<dataset_id>/`` (git-ignored) and checked against the
pinned revision and sha256 recorded here.
"""

from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Literal

from models.model import RegisteredObject

BehaviorKind = Literal["harmful", "benign"]

DEFAULT_BENCHMARK_ROOT = Path(__file__).resolve().parents[2] / "data" / "benchmarks"


@dataclass(frozen=True)
class Behavior:
    """One goal of a behavior dataset, normalized across datasets."""

    behavior_id: str
    goal: str
    target: str
    kind: BehaviorKind
    category: str
    behavior: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class BehaviorSplitSpec:
    """One file of a behavior dataset, pinned by hash."""

    kind: BehaviorKind
    filename: str
    sha256: str
    num_behaviors: int


@dataclass
class LLMDatasetSpec:
    dataset_id: str
    name: str
    hf_repo: str
    revision: str
    license: str
    gated: bool
    splits: dict[BehaviorKind, BehaviorSplitSpec]
    categories: tuple[str, ...]
    description: str | None = None
    objective: str = "jailbreak"
    # Normalized Behavior field -> CSV column
    field_map: dict[str, str] = field(default_factory=lambda: {
        "index": "Index",
        "goal": "Goal",
        "target": "Target",
        "category": "Category",
        "behavior": "Behavior",
        "source": "Source",
    })

    def default_root(self) -> Path:
        return DEFAULT_BENCHMARK_ROOT / self.dataset_id

    def info(self) -> dict[str, Any]:
        return {
            "id": self.dataset_id,
            "name": self.name,
            "description": self.description,
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "license": self.license,
            "gated": self.gated,
            "objective": self.objective,
            "categories": list(self.categories),
            "splits": {kind: split.num_behaviors for kind, split in self.splits.items()},
        }

    def to_registered_object(self) -> RegisteredObject:
        return RegisteredObject(
            id=self.dataset_id,
            name=self.name,
            description=self.description,
            parameters=[],
            task="language",
            objective=self.objective,
        )

    def load_behaviors(
            self,
            *,
            kinds: tuple[BehaviorKind, ...] = ("harmful", "benign"),
            categories: list[str] | None = None,
            subset: int | None = None,
            seed: int = 42,
            root: Path | None = None,
            verify_hash: bool = True,
    ) -> list[Behavior]:
        """Load the requested splits from the local copy of the dataset.

        ``subset`` is applied per kind and is balanced across categories, so a
        per-category ASR stays meaningful on small runs.
        """
        root = Path(root).expanduser() if root is not None else self.default_root()
        unknown = set(categories or []) - set(self.categories)
        if unknown:
            raise ValueError(f"Unknown categories for '{self.dataset_id}': {sorted(unknown)}")

        behaviors: list[Behavior] = []
        for kind in kinds:
            split = self.splits.get(kind)
            if split is None:
                raise ValueError(f"Dataset '{self.dataset_id}' has no '{kind}' split.")
            rows = self._read_split(split, root=root, verify_hash=verify_hash)
            if categories:
                rows = [row for row in rows if row.category in categories]
            if subset is not None:
                rows = _balanced_subset(rows, n=subset, seed=seed)
            behaviors.extend(rows)
        return behaviors

    def _read_split(self, split: BehaviorSplitSpec, *, root: Path, verify_hash: bool) -> list[Behavior]:
        path = root / split.filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Behavior file not found: {path}. Download it with: "
                f"hf download {self.hf_repo} --repo-type dataset "
                f"--revision {self.revision} --local-dir {root}"
            )
        if verify_hash:
            digest = _sha256(path)
            if digest != split.sha256:
                raise ValueError(
                    f"sha256 mismatch for {path}: expected {split.sha256}, got {digest}. "
                    f"The local copy is not revision {self.revision}."
                )

        fm = self.field_map
        with open(path, newline="", encoding="utf-8") as handle:
            return [
                Behavior(
                    behavior_id=f"{self.dataset_id}/{split.kind}/{row[fm['index']]}",
                    goal=row[fm["goal"]],
                    target=row[fm["target"]],
                    kind=split.kind,
                    category=row[fm["category"]],
                    behavior=row.get(fm["behavior"]),
                    source=row.get(fm["source"]),
                )
                for row in csv.DictReader(handle)
            ]


class AppLLMDatasetRegistry:
    """Concrete LLM dataset registry owned by nn_trust_apps."""

    def __init__(self) -> None:
        self._registry: dict[str, LLMDatasetSpec] = {}

    def register(self, spec: LLMDatasetSpec) -> None:
        self._registry[spec.dataset_id] = spec

    def list_specs(self) -> list[LLMDatasetSpec]:
        return list(self._registry.values())

    def get(self, dataset_id: str) -> LLMDatasetSpec:
        spec = self._registry.get(dataset_id)
        if spec is None:
            raise ValueError(f"Unknown LLM dataset '{dataset_id}'. Registered: {sorted(self._registry)}")
        return spec


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _balanced_subset(rows: list[Behavior], *, n: int, seed: int) -> list[Behavior]:
    """Round-robin over shuffled categories, then restore the dataset order."""
    rng = random.Random(seed)
    by_category: dict[str, list[Behavior]] = {}
    for row in rows:
        by_category.setdefault(row.category, []).append(row)
    queues = [rng.sample(items, len(items)) for _, items in sorted(by_category.items())]

    picked: list[Behavior] = []
    while len(picked) < n and any(queues):
        for queue in queues:
            if queue and len(picked) < n:
                picked.append(queue.pop())
    order = {row.behavior_id: i for i, row in enumerate(rows)}
    return sorted(picked, key=lambda row: order[row.behavior_id])


JBB_CATEGORIES = (
    "Disinformation",
    "Economic harm",
    "Expert advice",
    "Fraud/Deception",
    "Government decision-making",
    "Harassment/Discrimination",
    "Malware/Hacking",
    "Physical harm",
    "Privacy",
    "Sexual/Adult content",
)


def build_app_llm_dataset_registry() -> AppLLMDatasetRegistry:
    registry = AppLLMDatasetRegistry()
    registry.register(
        LLMDatasetSpec(
            dataset_id="jbb_behaviors",
            name="JailbreakBench Behaviors",
            hf_repo="JailbreakBench/JBB-Behaviors",
            revision="886acc352a31533ffbcf4ef22c744658688086fc",
            license="MIT",
            gated=False,
            splits={
                "harmful": BehaviorSplitSpec(
                    kind="harmful",
                    filename="data/harmful-behaviors.csv",
                    sha256="4a8ec6832056b631eb092dccc60d37a61c3d441268268888b3d006288afeffa1",
                    num_behaviors=100,
                ),
                "benign": BehaviorSplitSpec(
                    kind="benign",
                    filename="data/benign-behaviors.csv",
                    sha256="3cda234d21a991fa309bbfea4b6d9dae31ccdf8e9d452424b6a983e4fdc33468",
                    num_behaviors=100,
                ),
            },
            categories=JBB_CATEGORIES,
            description="100 harmful + 100 benign behaviors in 10 categories (JailbreakBench, NeurIPS'24 D&B).",
        )
    )
    return registry


@cache
def get_llm_dataset_registry() -> AppLLMDatasetRegistry:
    return build_app_llm_dataset_registry()


__all__ = [
    "AppLLMDatasetRegistry",
    "Behavior",
    "BehaviorKind",
    "BehaviorSplitSpec",
    "LLMDatasetSpec",
    "build_app_llm_dataset_registry",
    "get_llm_dataset_registry",
]
