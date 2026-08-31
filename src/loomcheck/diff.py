"""Comparing two runs.

This is the module the whole project earns its keep in. Everything before it exists to make
one question answerable: what changed, and is any of it worse?

It computes; `report.py` prints. Keeping those apart means the comparison can be tested
against hand-built runs without going near a terminal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loomcheck.graders import GRADER_NAMES
from loomcheck.models import Outcome, RunReport, ScenarioResult


@dataclass(frozen=True)
class MetricDelta:
    """One grader's pass rate, before and after.

    `None` means the grader never applied — `recovery` on a suite with no injected failures —
    and a rate that appears or disappears is reported as a change in coverage rather than as a
    delta, because there is no honest number to subtract.
    """

    name: str
    before: float | None
    after: float | None

    @property
    def delta(self) -> float | None:
        if self.before is None or self.after is None:
            return None
        return self.after - self.before

    @property
    def is_regression(self) -> bool:
        return self.delta is not None and self.delta < 0

    @property
    def is_improvement(self) -> bool:
        return self.delta is not None and self.delta > 0


@dataclass(frozen=True)
class GradeFlip:
    """A grader that changed its mind about one scenario."""

    grader: str
    was_passing: bool
    reason: str
    """The reason from the *later* run: what is true now, which is what needs acting on."""


@dataclass(frozen=True)
class ScenarioChange:
    """One scenario that behaved differently between the two runs."""

    scenario_id: str
    before_outcome: Outcome | None
    after_outcome: Outcome | None
    flips: list[GradeFlip] = field(default_factory=list)

    @property
    def outcome_changed(self) -> bool:
        return self.before_outcome is not self.after_outcome

    @property
    def regressed(self) -> bool:
        return any(flip.was_passing for flip in self.flips)


@dataclass(frozen=True)
class RunDiff:
    """Everything the diff command needs to print, and nothing about how to print it."""

    before: RunReport
    after: RunReport
    metrics: list[MetricDelta]
    changes: list[ScenarioChange]
    shared_scenarios: list[str]
    only_before: list[str]
    only_after: list[str]
    cost_before: float
    cost_after: float

    @property
    def regressions(self) -> list[MetricDelta]:
        return [m for m in self.metrics if m.is_regression]

    @property
    def improvements(self) -> list[MetricDelta]:
        return [m for m in self.metrics if m.is_improvement]

    @property
    def procedure_changed(self) -> bool:
        return self.before.procedure != self.after.procedure

    @property
    def model_changed(self) -> bool:
        """A model change alongside a procedure change makes the comparison unattributable."""
        return self.before.model != self.after.model

    @property
    def confounded(self) -> bool:
        return self.procedure_changed and self.model_changed

    @property
    def dashboard_would_be_green(self) -> bool:
        """True when something got worse while other numbers got better.

        The exact shape this tool exists to catch: an agent that stops blocking resolves more
        cases, faster, for less money, and every metric a team watches moves the right way.
        """
        cheaper = self.cost_after < self.cost_before
        return bool(self.regressions) and bool(self.improvements or cheaper)


def _pass_rate(results: list[ScenarioResult], grader: str) -> float | None:
    """Share of applicable gradings this grader passed, or None if it never applied."""
    grades = [g for r in results for g in r.grades if g.grader == grader and g.applicable]
    if not grades:
        return None
    return sum(1 for g in grades if g.passed) / len(grades)


def _by_id(report: RunReport) -> dict[str, ScenarioResult]:
    return {result.scenario_id: result for result in report.results}


def compare(before: RunReport, after: RunReport) -> RunDiff:
    """Compare two runs over the scenarios they have in common.

    Rates are computed over the shared set, not over each run's own results. Otherwise a suite
    that grew or shrank between runs would show a delta that mixes "the agent got worse" with
    "you ran different scenarios", and those are not the same finding. When the sets differ the
    report says so, because the numbers will then disagree with `loomcheck show`.
    """
    left, right = _by_id(before), _by_id(after)
    shared = sorted(left.keys() & right.keys())
    shared_before = [left[i] for i in shared]
    shared_after = [right[i] for i in shared]

    metrics = [
        MetricDelta(
            name=grader,
            before=_pass_rate(shared_before, grader),
            after=_pass_rate(shared_after, grader),
        )
        for grader in GRADER_NAMES
    ]

    changes: list[ScenarioChange] = []
    for scenario_id in shared:
        was, now = left[scenario_id], right[scenario_id]
        was_grades = {g.grader: g for g in was.grades}
        flips = [
            GradeFlip(grader=g.grader, was_passing=was_grades[g.grader].passed, reason=g.reason)
            for g in now.grades
            if g.grader in was_grades
            and g.applicable
            and was_grades[g.grader].applicable
            and g.passed != was_grades[g.grader].passed
        ]
        if flips or was.outcome is not now.outcome:
            changes.append(
                ScenarioChange(
                    scenario_id=scenario_id,
                    before_outcome=was.outcome,
                    after_outcome=now.outcome,
                    flips=flips,
                )
            )

    # Regressions first, then scenarios that merely changed shape. A reader scanning this list
    # is looking for what broke, not for what moved.
    changes.sort(key=lambda c: (not c.regressed, c.scenario_id))

    return RunDiff(
        before=before,
        after=after,
        metrics=metrics,
        changes=changes,
        shared_scenarios=shared,
        only_before=sorted(left.keys() - right.keys()),
        only_after=sorted(right.keys() - left.keys()),
        cost_before=_cost_per_case(shared_before),
        cost_after=_cost_per_case(shared_after),
    )


def _cost_per_case(results: list[ScenarioResult]) -> float:
    return sum(r.total_cost_usd for r in results) / len(results) if results else 0.0
