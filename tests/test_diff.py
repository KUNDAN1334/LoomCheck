"""Diff tests, against hand-built runs.

The diff is the output the project is judged on, so these check the arithmetic *and* the
judgement calls layered on top of it: what counts as a regression, what is left unflagged on
purpose, and when the comparison should refuse to attribute anything at all.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from loomcheck.diff import compare
from loomcheck.models import GradeResult, Outcome, RunReport, ScenarioResult

MODEL = "llama-3.3-70b-versatile"


def graded(
    scenario_id: str,
    outcome: Outcome | None,
    passes: dict[str, bool],
    *,
    cost: float = 0.0030,
    not_applicable: tuple[str, ...] = ("recovery",),
) -> ScenarioResult:
    grades = [
        GradeResult(
            grader=name,
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=f"{name} {'passed' if passed else 'failed'} on {scenario_id}",
        )
        for name, passed in passes.items()
    ]
    grades += [
        GradeResult(
            grader=name, applicable=False, passed=True, score=1.0, reason="nothing to check"
        )
        for name in not_applicable
        if name not in passes
    ]
    return ScenarioResult(
        scenario_id=scenario_id, outcome=outcome, grades=grades, total_cost_usd=cost
    )


def run(procedure: str, *results: ScenarioResult, model: str = MODEL) -> RunReport:
    return RunReport(
        run_id=uuid4(),
        started_at=datetime.now(UTC) - timedelta(hours=1),
        procedure=procedure,
        model=model,
        results=list(results),
    )


ALL_PASS = {"outcome": True, "trajectory": True, "escalation": True, "budget": True}


def test_a_grader_that_got_worse_is_flagged_and_one_that_improved_is_not() -> None:
    before = run(
        "claims_intake_v1",
        graded("a", Outcome.BLOCKED, ALL_PASS),
        graded("b", Outcome.APPROVE_CLAIM, {**ALL_PASS, "budget": False}),
    )
    after = run(
        "claims_intake_v2",
        graded("a", Outcome.APPROVE_CLAIM, {**ALL_PASS, "escalation": False, "outcome": False}),
        graded("b", Outcome.APPROVE_CLAIM, ALL_PASS),
    )

    diff = compare(before, after)
    by_name = {m.name: m for m in diff.metrics}

    assert by_name["escalation"].before == 1.0
    assert by_name["escalation"].after == 0.5
    assert by_name["escalation"].is_regression
    assert by_name["budget"].before == 0.5
    assert by_name["budget"].after == 1.0
    assert by_name["budget"].is_improvement
    assert not by_name["budget"].is_regression
    assert [m.name for m in diff.regressions] == ["outcome", "escalation"]


def test_a_changed_verdict_carries_the_later_reason() -> None:
    """What is true *now* is what needs acting on; the old reason describes a fixed past."""
    before = run("claims_intake_v1", graded("a", Outcome.BLOCKED, ALL_PASS))
    after = run(
        "claims_intake_v2",
        graded("a", Outcome.APPROVE_CLAIM, {**ALL_PASS, "escalation": False}),
    )

    change = compare(before, after).changes[0]
    assert change.scenario_id == "a"
    assert change.outcome_changed
    assert (change.before_outcome, change.after_outcome) == (
        Outcome.BLOCKED,
        Outcome.APPROVE_CLAIM,
    )
    assert [f.grader for f in change.flips] == ["escalation"]
    assert change.flips[0].was_passing is True
    assert change.flips[0].reason == "escalation failed on a"
    assert change.regressed


def test_a_grader_that_started_passing_is_recorded_as_a_change_but_not_a_regression() -> None:
    before = run("v1", graded("a", Outcome.APPROVE_CLAIM, {**ALL_PASS, "budget": False}))
    after = run("v2", graded("a", Outcome.APPROVE_CLAIM, ALL_PASS))

    change = compare(before, after).changes[0]
    assert change.flips[0].was_passing is False
    assert not change.regressed
    assert not change.outcome_changed


def test_regressed_scenarios_are_listed_first() -> None:
    """A reader scanning this list is looking for what broke, not for what moved."""
    before = run(
        "v1",
        graded("improved", Outcome.APPROVE_CLAIM, {**ALL_PASS, "budget": False}),
        graded("broke", Outcome.BLOCKED, ALL_PASS),
    )
    after = run(
        "v2",
        graded("improved", Outcome.APPROVE_CLAIM, ALL_PASS),
        graded("broke", Outcome.APPROVE_CLAIM, {**ALL_PASS, "escalation": False}),
    )
    assert [c.scenario_id for c in compare(before, after).changes] == ["broke", "improved"]


def test_an_unchanged_run_reports_nothing_changed() -> None:
    before = run("v1", graded("a", Outcome.APPROVE_CLAIM, ALL_PASS))
    after = run("v1", graded("a", Outcome.APPROVE_CLAIM, ALL_PASS))

    diff = compare(before, after)
    assert diff.changes == []
    assert diff.regressions == []
    assert not diff.procedure_changed
    assert not diff.dashboard_would_be_green


def test_rates_are_computed_over_the_shared_scenarios_only() -> None:
    """Otherwise a delta mixes 'the agent got worse' with 'you ran different scenarios', and
    those are not the same finding."""
    before = run(
        "v1",
        graded("shared", Outcome.APPROVE_CLAIM, ALL_PASS),
        graded("dropped", Outcome.APPROVE_CLAIM, {**ALL_PASS, "outcome": False}),
    )
    after = run(
        "v2",
        graded("shared", Outcome.APPROVE_CLAIM, ALL_PASS),
        graded("added", Outcome.APPROVE_CLAIM, {**ALL_PASS, "outcome": False}),
    )

    diff = compare(before, after)
    assert diff.shared_scenarios == ["shared"]
    assert diff.only_before == ["dropped"]
    assert diff.only_after == ["added"]
    # Both runs contain one failing scenario, but neither failure is in the shared set, so
    # outcome accuracy is 100% on both sides rather than 50%.
    outcome = next(m for m in diff.metrics if m.name == "outcome")
    assert (outcome.before, outcome.after) == (1.0, 1.0)


def test_a_model_change_alongside_a_procedure_change_is_flagged_as_unattributable() -> None:
    before = run("v1", graded("a", Outcome.APPROVE_CLAIM, ALL_PASS))
    after = run("v2", graded("a", Outcome.APPROVE_CLAIM, ALL_PASS), model="some-other-model")

    diff = compare(before, after)
    assert diff.model_changed and diff.procedure_changed
    assert diff.confounded


def test_the_same_procedure_on_two_models_is_not_confounded() -> None:
    """Comparing models on a fixed procedure is a legitimate thing to want."""
    before = run("v1", graded("a", Outcome.APPROVE_CLAIM, ALL_PASS))
    after = run("v1", graded("a", Outcome.APPROVE_CLAIM, ALL_PASS), model="other")
    assert not compare(before, after).confounded


def test_a_grader_that_starts_applying_shows_coverage_not_a_delta() -> None:
    """recovery goes from n/a to a real rate when failure scenarios enter the suite. There is
    no honest number to subtract from 'not applicable'."""
    before = run("v1", graded("a", Outcome.DENY_CLAIM, ALL_PASS))
    after = run("v1", graded("a", Outcome.DENY_CLAIM, {**ALL_PASS, "recovery": True}))

    recovery = next(m for m in compare(before, after).metrics if m.name == "recovery")
    assert recovery.before is None
    assert recovery.after == 1.0
    assert recovery.delta is None
    assert not recovery.is_regression


def test_cost_per_case_is_averaged_over_the_shared_set() -> None:
    before = run("v1", graded("a", Outcome.APPROVE_CLAIM, ALL_PASS, cost=0.004))
    after = run("v2", graded("a", Outcome.APPROVE_CLAIM, ALL_PASS, cost=0.002))
    diff = compare(before, after)
    assert (diff.cost_before, diff.cost_after) == (0.004, 0.002)


def test_the_green_dashboard_case_is_the_one_the_tool_exists_for() -> None:
    """Something got worse while the numbers a team watches got better."""
    before = run(
        "claims_intake_v1",
        graded("a", Outcome.BLOCKED, ALL_PASS, cost=0.004),
        graded("b", Outcome.APPROVE_CLAIM, {**ALL_PASS, "budget": False}, cost=0.004),
    )
    after = run(
        "claims_intake_v2",
        graded("a", Outcome.APPROVE_CLAIM, {**ALL_PASS, "escalation": False}, cost=0.002),
        graded("b", Outcome.APPROVE_CLAIM, ALL_PASS, cost=0.002),
    )

    diff = compare(before, after)
    assert diff.dashboard_would_be_green
    assert [m.name for m in diff.regressions] == ["escalation"]
    assert [m.name for m in diff.improvements] == ["budget"]
    assert diff.cost_after < diff.cost_before


def test_a_run_that_only_got_better_is_not_a_green_dashboard() -> None:
    """The line only means something when it is not printed every time."""
    before = run("v1", graded("a", Outcome.APPROVE_CLAIM, {**ALL_PASS, "budget": False}))
    after = run("v2", graded("a", Outcome.APPROVE_CLAIM, ALL_PASS))
    assert not compare(before, after).dashboard_would_be_green
