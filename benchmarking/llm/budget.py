"""Query counting for LLM benchmarks.

Every attack reaches the target through ``adapter.query(...)``. A counter is
attached to that method of one adapter instance, so the budget applies to all
attacks without changing them. Rules (roadmap, Phase 2):

1. one prompt sent to the target is one query: a batch of N prompts counts N,
   one multi-turn turn counts 1 even if the history is sent again;
2. hard limit: a call that would exceed the remaining budget is refused as a
   whole and every later call is refused too, so attacks that catch generic
   exceptions around a query cannot keep going past the budget;
3. attacker and judge calls are counted without a limit (run cost).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Optional

# Attack parameter whose value is the number of prompts sent to the target in
# the first batch. Attacks not listed send one prompt per call.
_FIRST_BATCH_PARAMETER: dict[str, str] = {
    "autodan": "population_size",
    "tree": "branching_factor",
    "treecrescendo": "branching_factor",
}


class QueryBudgetExceeded(Exception):
    """Raised by a counted adapter when a call would exceed the query budget."""


@dataclass
class CallCounter:
    """Counts prompts (or calls) and optionally enforces a hard budget."""

    budget: Optional[int] = None
    used: int = 0
    exhausted: bool = False

    def reset(self, budget: Optional[int] = None) -> None:
        self.budget, self.used, self.exhausted = budget, 0, False

    def consume(self, n: int) -> None:
        if self.exhausted or (self.budget is not None and self.used + n > self.budget):
            self.exhausted = True
            raise QueryBudgetExceeded(
                f"Query budget exhausted: {self.used}/{self.budget} used, a call of {n} prompt(s) was refused."
            )
        self.used += n


def count_prompts(messages: Any) -> int:
    """Number of prompts in a `query` call, with the batch rule of the NLP adapters."""
    is_batch = isinstance(messages, list) and len(messages) > 0 and not isinstance(messages[0], dict)
    return len(messages) if is_batch else 1


def attach_query_counter(adapter: Any, counter: CallCounter) -> Any:
    """Count (and limit) the prompts sent through `adapter.query` of this instance only."""
    query = adapter.query

    @wraps(query)
    def counted_query(messages, *args, **kwargs):
        counter.consume(count_prompts(messages))
        return query(messages, *args, **kwargs)

    adapter.query = counted_query
    return adapter


def attach_evaluate_counter(judge: Any, counter: CallCounter) -> Any:
    """Count the calls to `judge.evaluate` of this instance only."""
    evaluate = judge.evaluate

    @wraps(evaluate)
    def counted_evaluate(goal, response, *args, **kwargs):
        counter.consume(1)
        return evaluate(goal, response, *args, **kwargs)

    judge.evaluate = counted_evaluate
    return judge


def first_batch_size(attack_id: str, parameters: dict[str, Any]) -> int:
    """
    Prompts sent to the target by the first call of an attack.

    With a hard limit, an attack whose first batch exceeds the budget would
    never query the target and would silently score 0%.
    """
    name = _FIRST_BATCH_PARAMETER.get(attack_id)
    if name is None:
        return 1
    if parameters.get(name) is not None:
        return int(parameters[name])

    from nn_trust.attack import AttackFactory

    defaults = {param_id: field.default for param_id, field in AttackFactory.get_config_param(attack_id)}
    return int(defaults[name])


__all__ = [
    "CallCounter",
    "QueryBudgetExceeded",
    "attach_evaluate_counter",
    "attach_query_counter",
    "count_prompts",
    "first_batch_size",
]
