"""What a grader is, and the small amount of trace-reading they share.

One Protocol. No base class, no registry, no plugin surface: a grader is a function that
reads a scenario and a recorded trace and returns a verdict.

On scores: every grader returns 1.0 or 0.0 today, because every check here is a fact rather
than a degree — the agent either called a forbidden tool or it did not. Inventing fractional
scores would put a number on the report that nobody could act on. The field stays because a
grader that is genuinely continuous is plausible later, and the detail that would justify one
lives in `reason` instead, where it can be read.
"""

from __future__ import annotations

from typing import Protocol

from loomcheck.agent.tools import TERMINAL_OUTCOMES
from loomcheck.models import GradeResult, Scenario, ScenarioResult, TurnRecord


class Grader(Protocol):
    """A function from (scenario, recorded trace) to one verdict."""

    def __call__(self, scenario: Scenario, result: ScenarioResult) -> GradeResult: ...


def verdict(name: str, passed: bool, reason: str) -> GradeResult:
    return GradeResult(grader=name, passed=passed, score=1.0 if passed else 0.0, reason=reason)


def not_applicable(name: str, reason: str) -> GradeResult:
    """A grader with nothing to check.

    Kept out of the aggregate rather than passed. A pass rate inflated by graders that had
    nothing to grade is the same dishonest green this project exists to catch.
    """
    return GradeResult(grader=name, applicable=False, passed=True, score=1.0, reason=reason)


def terminal_turn(result: ScenarioResult) -> TurnRecord | None:
    """The turn that ended the case, if one did."""
    return next((t for t in result.turns if t.tool in TERMINAL_OUTCOMES), None)


def tool_calls(result: ScenarioResult) -> list[str]:
    return [turn.tool for turn in result.turns]
