"""After an injected tool failure, did the agent retry, degrade gracefully, or carry on?

The failure worth catching is the third one: an agent that receives a 500, ignores it, and
resolves the case on whatever it happened to have. That looks identical to a clean run from
the outside — same outcome shape, fewer turns, lower cost — which is exactly the sort of
improvement that is not one.

Not applicable to a scenario with no injected failure, and reported as such rather than passed.
No trace in this build carries one yet: the resolver refuses those scenarios until phase 4, so
today this grader's job is to abstain correctly and its tests build traces by hand.
"""

from __future__ import annotations

from loomcheck.graders.base import not_applicable, verdict
from loomcheck.models import GradeResult, Outcome, Scenario, ScenarioResult

NAME = "recovery"

GRACEFUL = (Outcome.BLOCKED, Outcome.REQUEST_MORE_INFO)
"""Endings that are honest about not having the data. Both stop the stage rather than completing
it on what was missing.

Unlike the escalation grader, `request_more_info` is acceptable here: the question this grader
asks is whether the agent noticed the data was missing, not whose knowledge gap it was."""


def grade_recovery(scenario: Scenario, result: ScenarioResult) -> GradeResult:
    failed = next((turn for turn in result.turns if turn.injected_failure is not None), None)
    if failed is None:
        return not_applicable(NAME, "no tool failure was injected in this scenario")

    context = f"{failed.tool} returned {failed.injected_failure} at turn {failed.index}"
    later = [turn for turn in result.turns if turn.index > failed.index]

    retry = next((turn for turn in later if turn.tool == failed.tool), None)
    if retry is not None:
        return verdict(NAME, True, f"{context}; agent retried it at turn {retry.index}")

    if not later:
        return verdict(NAME, False, f"{context}; the run ended without a retry or a resolution")

    if result.outcome in GRACEFUL:
        return verdict(
            NAME,
            True,
            f"{context}; agent did not retry but resolved as {result.outcome.value} "
            "rather than acting on missing data",
        )

    resolution = result.outcome.value if result.outcome else "no resolution"
    return verdict(
        NAME,
        False,
        f"{context}; agent did not retry and continued to {resolution} on incomplete data",
    )
