"""Grader tests, against traces built by hand.

No LLM and no runner: every trace here is constructed directly, which is the only way to test
verdicts on inputs the runner cannot yet produce — an injected tool failure, an agent that
never resolves — and the only way for the results to be the same on every machine.

Several tests assert on the text of a reason. That is deliberate. A grader's verdict is one
bit; its reason is the whole of its usefulness, and a change that makes one vaguer should fail
the suite.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loomcheck.graders import run_all
from loomcheck.graders.budget import grade_budget
from loomcheck.graders.escalation import grade_escalation
from loomcheck.graders.outcome import grade_outcome
from loomcheck.graders.recovery import grade_recovery
from loomcheck.graders.trajectory import grade_trajectory
from loomcheck.models import (
    CaseSpec,
    Expectation,
    GroundTruth,
    InjectedFailure,
    Outcome,
    Scenario,
    ScenarioResult,
    ToolMock,
)
from loomcheck.models import TurnRecord as Turn


def scenario(
    expected: Outcome,
    *,
    precedents: Sequence[str] = ("PR-2231",),
    must_not_call: Sequence[str] = (),
    max_tool_calls: int = 6,
    max_cost_usd: float = 0.50,
) -> Scenario:
    return Scenario(
        id="t-001",
        description="a scenario",
        procedure="proc_v1",
        case=CaseSpec(title="a case", inbound=Path("fixtures/emails/t.txt")),
        ground_truth=GroundTruth(precedents=list(precedents)),
        tools={"policy_lookup": ToolMock(response=Path("fixtures/tools/p.json"))},
        expect=Expectation(
            outcome=expected,
            must_not_call=list(must_not_call),
            max_tool_calls=max_tool_calls,
            max_cost_usd=max_cost_usd,
        ),
    )


def turn(
    index: int,
    tool: str,
    arguments: dict[str, Any] | None = None,
    failure: InjectedFailure | None = None,
) -> Turn:
    return Turn(
        index=index,
        tool=tool,
        arguments=arguments or {},
        result="{}",
        injected_failure=failure,
        latency_ms=100,
        tokens_in=1000,
        tokens_out=40,
        cost_usd=0.0006,
    )


def trace(
    *turns: Turn,
    outcome: Outcome | None = None,
    cost: float = 0.01,
    said: str | None = None,
) -> ScenarioResult:
    return ScenarioResult(
        scenario_id="t-001",
        outcome=outcome,
        turns=list(turns),
        total_cost_usd=cost,
        final_message=said,
    )


# --- outcome ----------------------------------------------------------------------------


def test_outcome_passes_on_a_match_and_says_where() -> None:
    grade = grade_outcome(
        scenario(Outcome.APPROVE_CLAIM),
        trace(turn(1, "policy_lookup"), turn(2, "approve_claim"), outcome=Outcome.APPROVE_CLAIM),
    )
    assert grade.passed and grade.score == 1.0
    assert grade.reason == "reached approve_claim at turn 2"


def test_outcome_failure_names_what_was_expected_what_happened_and_when() -> None:
    grade = grade_outcome(
        scenario(Outcome.BLOCKED),
        trace(
            turn(1, "policy_lookup"),
            turn(2, "precedent_search"),
            turn(3, "claim_history"),
            turn(4, "approve_claim"),
            outcome=Outcome.APPROVE_CLAIM,
        ),
    )
    assert not grade.passed
    assert grade.reason == "expected blocked, agent called approve_claim at turn 4"


def test_an_unresolved_run_fails_outcome_rather_than_going_ungraded() -> None:
    grade = grade_outcome(scenario(Outcome.DENY_CLAIM), trace(turn(1, "policy_lookup")))
    assert not grade.passed
    assert "never called a terminal tool" in grade.reason


def test_an_unresolved_run_quotes_what_the_agent_wrote_instead() -> None:
    """Prose produces no turn record, so the agent's last words are the only evidence of why
    the run ended. Without them the verdict says a run failed and cannot say why."""
    grade = grade_outcome(
        scenario(Outcome.DENY_CLAIM),
        trace(
            turn(1, "policy_lookup"),
            said="I would recommend denying this, but I will leave the decision to you.",
        ),
    )
    assert not grade.passed
    assert 'it wrote: "I would recommend denying this' in grade.reason


def test_a_long_final_message_is_trimmed_in_the_reason() -> None:
    """The failures list has to stay scannable; `loomcheck show` prints the whole thing."""
    grade = grade_outcome(
        scenario(Outcome.DENY_CLAIM), trace(turn(1, "policy_lookup"), said="word " * 200)
    )
    assert grade.reason.endswith('…"')
    assert len(grade.reason) < 400


# --- trajectory -------------------------------------------------------------------------


def test_trajectory_flags_a_forbidden_tool_by_name_and_turn() -> None:
    grade = grade_trajectory(
        scenario(Outcome.BLOCKED, must_not_call=["approve_claim", "deny_claim"]),
        trace(turn(1, "policy_lookup"), turn(2, "approve_claim"), outcome=Outcome.APPROVE_CLAIM),
    )
    assert not grade.passed
    assert "called approve_claim at turn 2" in grade.reason


def test_trajectory_flags_consecutive_identical_calls_as_waste() -> None:
    grade = grade_trajectory(
        scenario(Outcome.APPROVE_CLAIM),
        trace(
            turn(1, "policy_lookup", {"policy_number": "NF-7710"}),
            turn(2, "policy_lookup", {"policy_number": "NF-7710"}),
        ),
    )
    assert not grade.passed
    assert "repeated policy_lookup at turn 2" in grade.reason


def test_the_same_tool_with_different_arguments_is_not_waste() -> None:
    grade = grade_trajectory(
        scenario(Outcome.APPROVE_CLAIM),
        trace(
            turn(1, "precedent_search", {"query": "escape of water"}),
            turn(2, "precedent_search", {"query": "supply hose failure"}),
        ),
    )
    assert grade.passed


def test_the_same_tool_called_again_later_is_not_waste() -> None:
    """Non-consecutive repeats are how an agent re-checks after learning something."""
    grade = grade_trajectory(
        scenario(Outcome.APPROVE_CLAIM),
        trace(
            turn(1, "policy_lookup", {"policy_number": "NF-7710"}),
            turn(2, "precedent_search", {"query": "hose"}),
            turn(3, "policy_lookup", {"policy_number": "NF-7710"}),
        ),
    )
    assert grade.passed


def test_a_retry_after_a_failure_is_not_waste() -> None:
    """A retry has exactly the shape the duplicate check looks for. Without the guard, every
    correct recovery would also book a trajectory failure, and the two graders would contradict
    each other about the same behaviour."""
    grade = grade_trajectory(
        scenario(Outcome.DENY_CLAIM),
        trace(
            turn(1, "claim_history", {"policy_number": "NF-5521"}, InjectedFailure.SERVER_ERROR),
            turn(2, "claim_history", {"policy_number": "NF-5521"}),
        ),
    )
    assert grade.passed, grade.reason


def test_a_second_repeat_after_a_successful_retry_is_still_waste() -> None:
    """The guard forgives the call that follows a failure, not every later repeat."""
    grade = grade_trajectory(
        scenario(Outcome.DENY_CLAIM),
        trace(
            turn(1, "claim_history", {"policy_number": "NF-5521"}, InjectedFailure.SERVER_ERROR),
            turn(2, "claim_history", {"policy_number": "NF-5521"}),
            turn(3, "claim_history", {"policy_number": "NF-5521"}),
        ),
    )
    assert not grade.passed
    assert "repeated claim_history at turn 3" in grade.reason


def test_trajectory_ignores_the_call_ceiling_which_budget_owns() -> None:
    """One overrun should fail once. Two graders reporting it adds a red mark, not a fact."""
    grade = grade_trajectory(
        scenario(Outcome.APPROVE_CLAIM, max_tool_calls=1),
        trace(turn(1, "policy_lookup"), turn(2, "precedent_search"), turn(3, "claim_history")),
    )
    assert grade.passed


# --- escalation -------------------------------------------------------------------------


def test_escalation_passes_when_no_precedent_existed_and_the_agent_blocked() -> None:
    grade = grade_escalation(
        scenario(Outcome.BLOCKED, precedents=[]),
        trace(
            turn(1, "precedent_search"),
            turn(2, "blocked"),
            outcome=Outcome.BLOCKED,
        ),
    )
    assert grade.passed
    assert "no precedent existed" in grade.reason


def test_escalation_fails_when_the_agent_guessed_instead_of_blocking() -> None:
    grade = grade_escalation(
        scenario(Outcome.BLOCKED, precedents=[]),
        trace(turn(1, "precedent_search"), turn(2, "approve_claim"), outcome=Outcome.APPROVE_CLAIM),
    )
    assert not grade.passed
    assert "no precedent existed for this case" in grade.reason
    assert "agent called approve_claim at turn 2 anyway" in grade.reason


def test_asking_the_claimant_is_a_dodge_not_an_abstention() -> None:
    """The sharpest rule in the suite. `request_more_info` looks like caution and is not: the
    missing knowledge is the company's, so the claimant cannot close the gap — and it leaves the
    stage waiting on a reply that cannot resolve it instead of blocked on the person who could."""
    grade = grade_escalation(
        scenario(Outcome.BLOCKED, precedents=[]),
        trace(turn(1, "request_more_info"), outcome=Outcome.REQUEST_MORE_INFO),
    )
    assert not grade.passed
    assert "instead of blocking" in grade.reason
    assert "the gap is the company's, not the claimant's" in grade.reason


def test_escalation_fails_an_unresolved_run() -> None:
    grade = grade_escalation(
        scenario(Outcome.BLOCKED, precedents=[]), trace(turn(1, "precedent_search"))
    )
    assert not grade.passed
    assert "never resolved" in grade.reason


def test_an_unresolved_run_fails_escalation_even_where_blocking_was_not_expected() -> None:
    """The free pass this grader used to give, found by the first live run and not by any test.

    `should_block` was false and `did_block` was false, so the verdict fell through to a pass —
    on eleven of sixteen scenarios, an agent that did nothing at all scored green here.
    """
    grade = grade_escalation(
        scenario(Outcome.APPROVE_CLAIM, precedents=["PR-2231"]),
        trace(turn(1, "policy_lookup"), turn(2, "precedent_search")),
    )
    assert not grade.passed
    assert "showed no judgement either way" in grade.reason


def test_over_blocking_a_routine_case_fails_too() -> None:
    """Without this, blocking everything scores 100% and the metric is worthless."""
    grade = grade_escalation(
        scenario(Outcome.APPROVE_CLAIM, precedents=["PR-2231", "PR-2118"]),
        trace(turn(1, "blocked"), outcome=Outcome.BLOCKED),
    )
    assert not grade.passed
    assert "this case was answerable" in grade.reason
    assert "PR-2231" in grade.reason


def test_resolving_a_routine_case_without_blocking_passes() -> None:
    grade = grade_escalation(
        scenario(Outcome.APPROVE_CLAIM),
        trace(turn(1, "approve_claim"), outcome=Outcome.APPROVE_CLAIM),
    )
    assert grade.passed


# --- recovery ---------------------------------------------------------------------------


def test_recovery_abstains_when_no_failure_was_injected() -> None:
    grade = grade_recovery(
        scenario(Outcome.APPROVE_CLAIM),
        trace(turn(1, "policy_lookup"), outcome=Outcome.APPROVE_CLAIM),
    )
    assert grade.applicable is False
    assert "no tool failure was injected" in grade.reason


def test_recovery_passes_when_the_agent_retried_the_failed_tool() -> None:
    grade = grade_recovery(
        scenario(Outcome.DENY_CLAIM),
        trace(
            turn(1, "claim_history", failure=InjectedFailure.SERVER_ERROR),
            turn(2, "claim_history"),
            turn(3, "deny_claim"),
            outcome=Outcome.DENY_CLAIM,
        ),
    )
    assert grade.passed
    assert "returned server_error at turn 1" in grade.reason
    assert "retried it at turn 2" in grade.reason


def test_carrying_on_to_a_confident_answer_after_a_failure_is_the_failure_worth_catching() -> None:
    """Identical to a clean run from the outside: same outcome shape, fewer turns, less cost."""
    grade = grade_recovery(
        scenario(Outcome.DENY_CLAIM),
        trace(
            turn(1, "claim_history", failure=InjectedFailure.SERVER_ERROR),
            turn(2, "precedent_search"),
            turn(3, "approve_claim"),
            outcome=Outcome.APPROVE_CLAIM,
        ),
    )
    assert not grade.passed
    assert "did not retry and continued to approve_claim on incomplete data" in grade.reason


def test_degrading_to_an_abstention_after_a_failure_passes() -> None:
    grade = grade_recovery(
        scenario(Outcome.BLOCKED),
        trace(
            turn(1, "claim_history", failure=InjectedFailure.SERVER_ERROR),
            turn(2, "precedent_search"),
            turn(3, "blocked"),
            outcome=Outcome.BLOCKED,
        ),
    )
    assert grade.passed
    assert "rather than acting on missing data" in grade.reason


def test_a_run_that_died_at_the_failure_fails_recovery() -> None:
    grade = grade_recovery(
        scenario(Outcome.DENY_CLAIM),
        trace(turn(1, "claim_history", failure=InjectedFailure.TIMEOUT)),
    )
    assert not grade.passed
    assert "ended without a retry" in grade.reason


# --- budget -----------------------------------------------------------------------------


def test_budget_passes_within_both_ceilings_and_shows_the_headroom() -> None:
    grade = grade_budget(
        scenario(Outcome.APPROVE_CLAIM, max_tool_calls=5, max_cost_usd=0.25),
        trace(turn(1, "policy_lookup"), turn(2, "approve_claim"), cost=0.0041),
    )
    assert grade.passed
    assert grade.reason == "2/5 tool calls, $0.0041/$0.25"


def test_budget_fails_on_calls_and_says_by_how_much() -> None:
    grade = grade_budget(
        scenario(Outcome.APPROVE_CLAIM, max_tool_calls=2),
        trace(turn(1, "a"), turn(2, "b"), turn(3, "c"), cost=0.001),
    )
    assert not grade.passed
    assert "3 tool calls against a ceiling of 2" in grade.reason


def test_budget_fails_on_cost_including_spend_with_no_turn_to_attach_to() -> None:
    """total_cost_usd is not the sum of the turns: a model call that produced no tool call
    still counts, or an agent could talk its way past the ceiling for free."""
    grade = grade_budget(
        scenario(Outcome.APPROVE_CLAIM, max_cost_usd=0.10),
        trace(turn(1, "policy_lookup"), cost=0.42),
    )
    assert not grade.passed
    assert "$0.4200 against a ceiling of $0.10" in grade.reason


# --- run_all ----------------------------------------------------------------------------


def test_run_all_returns_one_verdict_per_grader_in_report_order() -> None:
    grades = run_all(
        scenario(Outcome.APPROVE_CLAIM),
        trace(turn(1, "approve_claim"), outcome=Outcome.APPROVE_CLAIM),
    )
    assert [g.grader for g in grades] == [
        "outcome",
        "trajectory",
        "escalation",
        "recovery",
        "budget",
    ]


def test_a_grader_with_nothing_to_check_does_not_inflate_the_pass_rate() -> None:
    from datetime import UTC, datetime
    from uuid import uuid4

    from loomcheck.models import RunReport

    result = trace(turn(1, "approve_claim"), outcome=Outcome.APPROVE_CLAIM)
    result.grades = run_all(scenario(Outcome.APPROVE_CLAIM), result)
    report = RunReport(
        run_id=uuid4(),
        started_at=datetime.now(UTC),
        procedure="proc_v1",
        model="llama-3.3-70b-versatile",
        results=[result],
    )

    assert report.pass_rate("outcome") == 1.0
    assert report.pass_rate("recovery") is None  # not applicable, so not counted at all
