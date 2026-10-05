from typing import Any, Literal, Optional, Self

from pydantic import BaseModel, Field, model_validator

from models.info import ModelInfo
from models.model import RegisteredObject

# Attacks that cannot produce a judged target response in this benchmark.
UNSUPPORTED_LLM_BENCHMARK_ATTACKS: dict[str, str] = {
    "gcg": "white-box suffix optimization: it never generates a target response "
           "(best_response is the adversarial suffix) and needs model weights.",
}


class LLMTargetSpec(BaseModel):
    """
    Target model and the generation settings that are part of the benchmark protocol.
    """
    model: ModelInfo
    system_prompt: Optional[str] = None
    temperature: float = Field(default=0.0, ge=0.0)
    max_new_tokens: int = Field(default=512, gt=0)
    n_ctx: Optional[int] = Field(default=None, gt=0)
    seed: int = 0


class LLMBehaviorSelection(BaseModel):
    """
    Which behaviors of a registered dataset are used.
    """
    id: str = "jbb_behaviors"
    categories: Optional[list[str]] = None
    subset: Optional[int] = Field(
        default=None,
        gt=0,
        description="Behaviors per kind (harmful/benign), balanced across categories."
    )
    seed: int = 42
    include_benign: bool = Field(
        default=True,
        description="Benign behaviors are asked without attack, to measure over-refusal."
    )


class LLMAttackSelection(BaseModel):
    attack: RegisteredObject
    query_budget: Optional[int] = Field(
        default=None,
        gt=0,
        description="Overrides the run budget for this attack: its ASR is then not budget-matched."
    )
    seeds: list[int] = Field(default_factory=lambda: [0], min_length=1)


class LLMJudgeSelection(BaseModel):
    id: str
    threshold: Optional[float] = None
    model: Optional[ModelInfo] = Field(
        default=None,
        description="Overrides the registry model (e.g. the non quantized build on the server)."
    )


class LLMMetricSelection(BaseModel):
    id: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class LLMBenchmarkOptions(BaseModel):
    output_path: str = "~/Desktop/StableAI/benchmark_repository"
    gpu: bool = True
    resume: bool = True
    store_responses: bool = True
    split_phases: bool = Field(
        default=True,
        description="Generate every response first, then load the benchmark judges one at a time."
    )
    on_error: Literal["skip", "abort"] = "skip"
    verbose: bool = True
    create_pdf: bool = False


class LLMBenchmarkConfig(BaseModel):
    """
    This class is for handling the input of the LLM benchmark service.
    Registry ids (dataset, judges, metrics) are checked against the app registries.
    """
    target: LLMTargetSpec
    attacker: Optional[ModelInfo] = None
    attack_judge: LLMJudgeSelection = Field(default_factory=lambda: LLMJudgeSelection(id="jailjudge"))
    dataset: LLMBehaviorSelection = Field(default_factory=LLMBehaviorSelection)
    attacks: list[LLMAttackSelection] = Field(default_factory=list)
    query_budget: int = Field(
        gt=0,
        description="Maximum number of prompts sent to the target per behavior, for every attack (hard limit)."
    )
    benchmark_judges: list[LLMJudgeSelection] = Field(min_length=1)
    metrics: list[LLMMetricSelection] = Field(min_length=1)
    options: LLMBenchmarkOptions = Field(default_factory=LLMBenchmarkOptions)
    benchmark_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_attacker(self) -> Self:
        # Same default as JailbreakAttackProps: the target attacks itself.
        if self.attacker is None:
            self.attacker = self.target.model
        return self

    @model_validator(mode="after")
    def validate_attacks(self) -> Self:
        ids = [selection.attack.id for selection in self.attacks]
        duplicates = sorted({attack_id for attack_id in ids if ids.count(attack_id) > 1})
        if duplicates:
            raise ValueError(f"Attacks selected more than once: {duplicates}")
        for selection in self.attacks:
            reason = UNSUPPORTED_LLM_BENCHMARK_ATTACKS.get(selection.attack.id)
            if reason is not None:
                raise ValueError(f"Attack '{selection.attack.id}' is not supported by the LLM benchmark: {reason}")
            objective = selection.attack.objective
            if objective is not None and objective != "jailbreak":
                raise ValueError(
                    f"Attack '{selection.attack.id}' has objective '{objective}', "
                    "the LLM benchmark only supports 'jailbreak'."
                )
        return self

    @model_validator(mode="after")
    def validate_registries(self) -> Self:
        # Imported here: `benchmarking` imports `models` at package level.
        from benchmarking.llm import get_llm_dataset_registry, get_llm_judge_registry, get_llm_metric_registry

        dataset = get_llm_dataset_registry().get(self.dataset.id)
        unknown = set(self.dataset.categories or []) - set(dataset.categories)
        if unknown:
            raise ValueError(f"Unknown categories for '{dataset.dataset_id}': {sorted(unknown)}")
        if self.dataset.include_benign and "benign" not in dataset.splits:
            raise ValueError(f"Dataset '{dataset.dataset_id}' has no benign split.")
        if self.dataset.subset is not None:
            smallest = min(split.num_behaviors for split in dataset.splits.values())
            if self.dataset.subset > smallest:
                raise ValueError(f"subset={self.dataset.subset} exceeds the split size ({smallest}).")

        judges = get_llm_judge_registry()
        for selection, role in (
                (self.attack_judge, "attack"),
                *((judge, "benchmark") for judge in self.benchmark_judges),
        ):
            spec = judges.get(selection.id)
            if role not in spec.roles:
                raise ValueError(f"Judge '{spec.judge_id}' cannot be used as {role} judge (roles: {sorted(spec.roles)}).")
            if spec.model_id is None and selection.model is None:
                raise ValueError(f"Judge '{spec.judge_id}' has no local build: pass `model` explicitly.")
        judge_ids = [judge.id for judge in self.benchmark_judges]
        if len(set(judge_ids)) != len(judge_ids):
            raise ValueError(f"Benchmark judges selected more than once: {judge_ids}")

        metrics = get_llm_metric_registry()
        metric_ids = [metric.id for metric in self.metrics]
        if len(set(metric_ids)) != len(metric_ids):
            raise ValueError(f"Metrics selected more than once: {metric_ids}")
        for selection in self.metrics:
            spec = metrics.get(selection.id)
            unknown = set(selection.parameters) - set(spec.parameter_defaults())
            if unknown:
                raise ValueError(f"Unknown parameters for metric '{spec.metric_id}': {sorted(unknown)}")
            if spec.requires.benign and not self.dataset.include_benign:
                raise ValueError(f"Metric '{spec.metric_id}' needs the benign behaviors (include_benign=True).")
            if len(self.benchmark_judges) < spec.requires.min_benchmark_judges:
                raise ValueError(
                    f"Metric '{spec.metric_id}' needs at least "
                    f"{spec.requires.min_benchmark_judges} benchmark judges."
                )
            if spec.requires.attacks and not self.attacks:
                raise ValueError(f"Metric '{spec.metric_id}' needs at least one attack.")
        return self

    def metric_parameters(self, metric_id: str) -> dict[str, Any]:
        """
        Registry defaults overridden by the selected parameters.
        """
        from benchmarking.llm import get_llm_metric_registry

        selection = next(metric for metric in self.metrics if metric.id == metric_id)
        return get_llm_metric_registry().get(metric_id).parameter_defaults() | selection.parameters

    def budget_for(self, attack_id: str) -> int:
        """
        Query budget of one attack: its override, otherwise the run budget.
        """
        selection = next(attack for attack in self.attacks if attack.attack.id == attack_id)
        return selection.query_budget or self.query_budget

    def protocol_warnings(self) -> list[str]:
        """
        Valid but weak choices, to be recorded with the results.
        """
        from benchmarking.llm import get_llm_judge_registry

        judges = get_llm_judge_registry()
        warnings: list[str] = []
        if [judge.id for judge in self.benchmark_judges] == [self.attack_judge.id]:
            warnings.append(
                f"'{self.attack_judge.id}' both guides the attacks and measures the ASR: "
                "the ASR may be inflated. Add a control judge."
            )
        for selection in self.benchmark_judges:
            if judges.get(selection.id).goal_conditioned is False:
                warnings.append(f"Benchmark judge '{selection.id}' is not goal-conditioned.")
        for selection in self.attacks:
            if self.budget_for(selection.attack.id) != self.query_budget:
                warnings.append(
                    f"Attack '{selection.attack.id}' overrides the run budget "
                    f"({selection.query_budget} vs {self.query_budget}): its ASR is not budget-matched."
                )
        return warnings
