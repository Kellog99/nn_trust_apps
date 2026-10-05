"""App-owned registry of LLM benchmark metrics.

A spec declares what a metric needs (benign split, number of judges, attacks)
so a configuration can be rejected before any model is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import Any, Literal

from models.model import ParametersProps, RegisteredObject

MetricOutput = Literal["scalar", "per_category", "per_budget", "matrix"]


@dataclass(frozen=True)
class MetricRequirements:
    benign: bool = False
    min_benchmark_judges: int = 1
    attacks: bool = False


@dataclass
class LLMMetricSpec:
    metric_id: str
    name: str
    description: str
    output: MetricOutput
    requires: MetricRequirements = field(default_factory=MetricRequirements)
    parameters: list[ParametersProps] = field(default_factory=list)

    def parameter_defaults(self) -> dict[str, Any]:
        return {param.id: param.default for param in self.parameters}

    def info(self) -> dict[str, Any]:
        return {
            "id": self.metric_id,
            "name": self.name,
            "description": self.description,
            "output": self.output,
            "requires": vars(self.requires),
            "parameters": [param.model_dump() for param in self.parameters],
        }

    def to_registered_object(self) -> RegisteredObject:
        return RegisteredObject(
            id=self.metric_id,
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            task="language",
            objective="jailbreak",
        )


class AppLLMMetricRegistry:
    """Concrete LLM metric registry owned by nn_trust_apps."""

    def __init__(self) -> None:
        self._registry: dict[str, LLMMetricSpec] = {}

    def register(self, spec: LLMMetricSpec) -> None:
        self._registry[spec.metric_id] = spec

    def list_specs(self) -> list[LLMMetricSpec]:
        return list(self._registry.values())

    def get(self, metric_id: str) -> LLMMetricSpec:
        spec = self._registry.get(metric_id)
        if spec is None:
            raise ValueError(f"Unknown LLM metric '{metric_id}'. Registered: {sorted(self._registry)}")
        return spec


_CI_LEVEL = ParametersProps(
    id="ci_level",
    name="Confidence level",
    min=0.5,
    max=0.99,
    step=0.01,
    default=0.95,
    description="Level of the Wilson confidence interval.",
    kind="number",
)


def build_app_llm_metric_registry() -> AppLLMMetricRegistry:
    registry = AppLLMMetricRegistry()
    registry.register(LLMMetricSpec(
        metric_id="asr",
        name="Attack Success Rate",
        description="Share of harmful behaviors judged jailbroken, with Wilson confidence interval.",
        output="scalar",
        parameters=[_CI_LEVEL],
    ))
    registry.register(LLMMetricSpec(
        metric_id="asr_by_category",
        name="ASR by category",
        description="ASR for each harm category of the dataset.",
        output="per_category",
        parameters=[_CI_LEVEL],
    ))
    registry.register(LLMMetricSpec(
        metric_id="asr_at_budget",
        name="ASR at query budget",
        description="ASR counting only successes reached within k queries to the target.",
        output="per_budget",
        requires=MetricRequirements(attacks=True),
        parameters=[
            ParametersProps(
                id="k",
                name="Budget",
                min=1,
                step=1,
                default=20,
                description="Maximum number of queries to the target.",
                kind="number",
            ),
            _CI_LEVEL,
        ],
    ))
    registry.register(LLMMetricSpec(
        metric_id="refusal_benign",
        name="Over-refusal",
        description="Share of benign behaviors the target refuses, asked without attack.",
        output="scalar",
        requires=MetricRequirements(benign=True),
        parameters=[_CI_LEVEL],
    ))
    registry.register(LLMMetricSpec(
        metric_id="resilience_gap",
        name="Resilience Gap",
        description="Safety without attack minus safety under attack (MLCommons AILuminate).",
        output="scalar",
        requires=MetricRequirements(attacks=True),
    ))
    registry.register(LLMMetricSpec(
        metric_id="judge_agreement",
        name="Judge agreement",
        description="Pairwise Cohen's kappa between the benchmark judges.",
        output="matrix",
        requires=MetricRequirements(min_benchmark_judges=2),
    ))
    registry.register(LLMMetricSpec(
        metric_id="mean_queries",
        name="Mean queries to success",
        description="Mean number of queries to the target before the first success.",
        output="scalar",
        requires=MetricRequirements(attacks=True),
    ))
    return registry


@cache
def get_llm_metric_registry() -> AppLLMMetricRegistry:
    return build_app_llm_metric_registry()


__all__ = [
    "AppLLMMetricRegistry",
    "LLMMetricSpec",
    "MetricOutput",
    "MetricRequirements",
    "build_app_llm_metric_registry",
    "get_llm_metric_registry",
]
