"""Integrity of the shipped scenario suite.

A grader is only as honest as the cases it is pointed at. These tests do not check the agent
or the harness — they check that the suite itself has not drifted into something a bad agent
could score well on, which is a failure mode no amount of correct code prevents.
"""

from __future__ import annotations

from collections import Counter

from loomcheck.config import PROJECT_ROOT
from loomcheck.graders import run_all
from loomcheck.loader import load_scenarios
from loomcheck.models import InjectedFailure, Outcome, Scenario
from loomcheck.runner import execute_run
from tests.stubs import ScriptedChatModel, tool_call

CLAIMS = PROJECT_ROOT / "scenarios" / "claims"
MODEL = "llama-3.3-70b-versatile"


def suite() -> list[Scenario]:
    return load_scenarios(CLAIMS)


def test_the_suite_is_the_size_the_project_committed_to() -> None:
    """12-15 was the original scope; the sixteenth was added deliberately — see D-047."""
    assert 12 <= len(suite()) <= 16


def test_every_outcome_appears_more_than_once() -> None:
    """A resolution the suite tests once is a resolution one lucky guess covers."""
    counts = Counter(s.expect.outcome for s in suite())
    assert set(counts) == set(Outcome)
    assert all(count >= 2 for count in counts.values()), counts


def test_answerable_cases_outnumber_escalations_by_at_least_two_to_one() -> None:
    """The escalation grader is the one worth gaming: an agent that hands everything over
    scores 100% on recall. Precision has to be tested harder than recall or the number lies."""
    counts = Counter(s.expect.outcome for s in suite())
    escalations = counts[Outcome.BLOCKED]
    answerable = sum(counts.values()) - escalations
    assert escalations >= 3, "too few escalations to measure recall at all"
    assert answerable >= 2 * escalations, (escalations, answerable)


def test_a_missing_precedent_and_an_expected_escalation_are_the_same_thing() -> None:
    """The one coherence rule the loader deliberately does not enforce, pinned here instead.

    A scenario declaring no precedent while expecting a confident answer — or the reverse —
    would make the escalation grader's reason string describe a case that does not exist.
    """
    for scenario in suite():
        expects_escalation = scenario.expect.outcome is Outcome.BLOCKED
        assert scenario.requires_escalation == expects_escalation, scenario.id


def test_no_scenario_forbids_the_outcome_it_expects() -> None:
    for scenario in suite():
        assert scenario.expect.outcome.value not in scenario.expect.must_not_call, scenario.id


def test_every_failure_mode_is_exercised_somewhere() -> None:
    """Four modes were built. A mode with no scenario behind it is untested code."""
    used = {mock.failure for s in suite() for mock in s.tools.values() if mock.failure}
    assert used == set(InjectedFailure), set(InjectedFailure) - used


def test_the_suite_contains_both_a_recoverable_failure_and_a_permanent_outage() -> None:
    """They call for opposite behaviour — retry, versus stop and hand over — so a suite with
    only one of them rewards an agent that always does the same thing."""
    mocks = [mock for s in suite() for mock in s.tools.values() if mock.failure]
    assert any(mock.then is not None for mock in mocks)
    assert any(mock.then is None for mock in mocks)


def test_roughly_a_third_of_scenarios_exercise_recovery() -> None:
    """Enough that recovery's pass rate moves on more than one case, not so many that the suite
    is about tool failures rather than about judgement."""
    with_failures = [s for s in suite() if any(m.failure for m in s.tools.values())]
    assert 3 <= len(with_failures) <= len(suite()) // 2


def test_every_scenario_executes_and_grades() -> None:
    """A smoke test over the whole suite: one generic script, run against all of it.

    The agent it simulates is a poor one — it never retries and escalates everything — so most
    scenarios fail. That is the point: what is being checked is that all fifteen run to a
    verdict rather than raising, which is what a missing fixture or an undeclared tool would do.
    """
    scenarios = suite()
    script = [
        tool_call("policy_lookup", {"policy_number": "NF-0000"}),
        tool_call("precedent_search", {"query": "the loss"}),
        tool_call("claim_history", {"policy_number": "NF-0000"}),
        tool_call("blocked", {"reason": "unfamiliar"}),
    ]
    report = execute_run(
        scenarios=scenarios,
        model=ScriptedChatModel(script=script),
        model_name=MODEL,
        procedure="claims_intake_v1",
    )

    assert len(report.results) == len(scenarios)
    assert all(len(r.grades) == 5 for r in report.results)
    assert all(r.outcome is Outcome.BLOCKED for r in report.results)


def test_an_agent_that_escalates_everything_scores_badly() -> None:
    """The suite's whole defence against a gamed metric, asserted rather than assumed."""
    scenarios = suite()
    escalate_always = [
        tool_call("policy_lookup", {"policy_number": "NF-0000"}),
        tool_call("blocked", {"reason": "unfamiliar"}),
    ]
    report = execute_run(
        scenarios=scenarios,
        model=ScriptedChatModel(script=escalate_always),
        model_name=MODEL,
        procedure="claims_intake_v1",
    )
    for scenario in scenarios:
        result = next(r for r in report.results if r.scenario_id == scenario.id)
        result.grades = run_all(scenario, result)

    escalation = report.pass_rate("escalation")
    assert escalation is not None and escalation < 0.35, escalation


def test_every_scenario_runs_against_both_procedures() -> None:
    """The diff compares v1 to v2 over this suite, so v2 has to be runnable against all of it."""
    scenarios = suite()
    script = [tool_call("blocked", {"reason": "unfamiliar"})]
    report = execute_run(
        scenarios=scenarios,
        model=ScriptedChatModel(script=script),
        model_name=MODEL,
        procedure="claims_intake_v2",
    )
    assert len(report.results) == len(scenarios)
