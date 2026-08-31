"""Within the call and cost ceilings?

Owns both ceilings, so an overrun fails once and says by how much. Cost is summed from the
recorded turns plus any model call that produced no tool call — the agent writing prose still
spends money, and a ceiling that ignored it could be talked past for free.

A retry after an injected failure counts as a real call here, because it is one. Recovery
judges whether retrying was right; budget only counts what it cost. That interaction is why
`claims-fr-002` allows 7 calls where the others allow 5 and 6.
"""

from __future__ import annotations

from loomcheck.graders.base import verdict
from loomcheck.models import GradeResult, Scenario, ScenarioResult

NAME = "budget"


def grade_budget(scenario: Scenario, result: ScenarioResult) -> GradeResult:
    calls = len(result.turns)
    max_calls = scenario.expect.max_tool_calls
    cost = result.total_cost_usd
    max_cost = scenario.expect.max_cost_usd

    problems: list[str] = []
    if calls > max_calls:
        problems.append(f"{calls} tool calls against a ceiling of {max_calls}")
    if cost > max_cost:
        problems.append(f"${cost:.4f} against a ceiling of ${max_cost:.2f}")

    if problems:
        return verdict(NAME, False, "over budget: " + "; ".join(problems))

    return verdict(
        NAME,
        True,
        f"{calls}/{max_calls} tool calls, ${cost:.4f}/${max_cost:.2f}",
    )
