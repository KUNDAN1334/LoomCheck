"""Runner tests: the whole path from a scenario file to a recorded trace, minus the LLM.

These drive the real graph, the real resolver and the real recorder against a scripted model,
so what they exercise is the wiring — which is the part that can be wrong without anyone
noticing until a suite has already been paid for.
"""

from __future__ import annotations

import pytest

from loomcheck.agent.graph import ParallelToolCallError
from loomcheck.config import PROJECT_ROOT
from loomcheck.graders import run_all
from loomcheck.loader import load_scenario
from loomcheck.mocks.resolver import UndeclaredToolError
from loomcheck.models import InjectedFailure, Outcome
from loomcheck.runner import MAX_TURNS, execute_run, execute_scenario
from tests.stubs import ScriptedChatModel, parallel_tool_calls, prose, tool_call

CLAIMS = PROJECT_ROOT / "scenarios" / "claims"
MODEL = "llama-3.3-70b-versatile"

HANDLED_WELL = [
    tool_call("policy_lookup", {"policy_number": "NF-7710"}),
    tool_call("precedent_search", {"query": "escape of water, failed supply hose"}),
    tool_call("claim_history", {"policy_number": "NF-7710"}),
    tool_call("approve_claim", {"amount_eur": 4395.0, "reason": "PR-2231 is on all fours"}),
]


def test_a_resolved_scenario_records_every_turn_in_order() -> None:
    scenario = load_scenario(CLAIMS / "claims-wd-001.yaml")
    result = execute_scenario(scenario, ScriptedChatModel(script=HANDLED_WELL), MODEL)

    assert result.outcome is Outcome.APPROVE_CLAIM
    assert [t.tool for t in result.turns] == [
        "policy_lookup",
        "precedent_search",
        "claim_history",
        "approve_claim",
    ]
    assert result.total_cost_usd > 0


def test_the_agent_sees_the_fixture_not_the_scenario_file() -> None:
    scenario = load_scenario(CLAIMS / "claims-wd-001.yaml")
    result = execute_scenario(scenario, ScriptedChatModel(script=HANDLED_WELL), MODEL)
    assert "NF-7710" in result.turns[0].result
    assert "PR-2231" in result.turns[1].result


def test_a_terminal_tool_ends_the_run_even_with_script_left_over() -> None:
    scenario = load_scenario(CLAIMS / "claims-wd-001.yaml")
    script = [tool_call("blocked", {"reason": "unfamiliar"}), *HANDLED_WELL]
    result = execute_scenario(scenario, ScriptedChatModel(script=script), MODEL)

    assert result.outcome is Outcome.BLOCKED
    assert len(result.turns) == 1


def test_an_agent_that_never_resolves_stops_at_the_turn_ceiling() -> None:
    """An unresolved run is a recorded failure, not a crash and not an infinite bill."""
    scenario = load_scenario(CLAIMS / "claims-wd-001.yaml")
    script = [tool_call("policy_lookup", {"policy_number": "NF-7710"})]
    result = execute_scenario(scenario, ScriptedChatModel(script=script), MODEL)

    assert result.outcome is None
    assert len(result.turns) == MAX_TURNS


def test_prose_with_no_tool_call_ends_the_run_but_still_costs() -> None:
    scenario = load_scenario(CLAIMS / "claims-wd-001.yaml")
    result = execute_scenario(scenario, ScriptedChatModel(script=[prose()]), MODEL)

    assert result.outcome is None
    assert result.turns == []
    assert result.total_cost_usd > 0


def test_calling_a_tool_the_scenario_does_not_mock_fails_the_run() -> None:
    scenario = load_scenario(CLAIMS / "claims-wd-001.yaml")
    script = [tool_call("weather_lookup", {"city": "Berlin"})]
    with pytest.raises(UndeclaredToolError):
        execute_scenario(scenario, ScriptedChatModel(script=script), MODEL)


def test_parallel_tool_calls_are_refused_rather_than_silently_truncated() -> None:
    scenario = load_scenario(CLAIMS / "claims-wd-001.yaml")
    script = [parallel_tool_calls("policy_lookup", "claim_history")]
    with pytest.raises(ParallelToolCallError) as exc:
        execute_scenario(scenario, ScriptedChatModel(script=script), MODEL)
    assert "one tool call per turn" in str(exc.value)


def test_a_failed_call_is_recorded_and_the_retry_is_a_turn_of_its_own() -> None:
    """The whole chain: scenario file, injected failure, recovery, recorded trace, grade."""
    scenario = load_scenario(CLAIMS / "claims-fr-002.yaml")
    script = [
        tool_call("policy_lookup", {"policy_number": "NF-5521"}),
        tool_call("precedent_search", {"query": "kitchen fire, lapsed policy"}),
        tool_call("claim_history", {"policy_number": "NF-5521"}),  # 500
        tool_call("claim_history", {"policy_number": "NF-5521"}),  # retry, succeeds
        tool_call("deny_claim", {"reason": "cover ceased 2026-06-01 for non-payment"}),
    ]
    result = execute_scenario(scenario, ScriptedChatModel(script=script), MODEL)

    assert result.outcome is Outcome.DENY_CLAIM
    assert result.turns[2].injected_failure is InjectedFailure.SERVER_ERROR
    assert result.turns[3].injected_failure is None
    assert "CL-2025-1188" in result.turns[3].result

    grades = {g.grader: g for g in run_all(scenario, result)}
    assert grades["recovery"].applicable and grades["recovery"].passed
    assert "retried it at turn 4" in grades["recovery"].reason
    # 5 calls including the retry, against this scenario's deliberately looser ceiling of 7.
    assert grades["budget"].passed
    assert all(g.passed for g in grades.values())


def test_carrying_on_after_a_failure_fails_recovery_on_a_real_trace() -> None:
    scenario = load_scenario(CLAIMS / "claims-fr-002.yaml")
    script = [
        tool_call("policy_lookup", {"policy_number": "NF-5521"}),
        tool_call("claim_history", {"policy_number": "NF-5521"}),  # 500, never retried
        tool_call("deny_claim", {"reason": "lapsed"}),
    ]
    result = execute_scenario(scenario, ScriptedChatModel(script=script), MODEL)

    grades = {g.grader: g for g in run_all(scenario, result)}
    assert grades["outcome"].passed  # right answer...
    assert not grades["recovery"].passed  # ...reached on data it never actually got
    assert "did not retry and continued to deny_claim" in grades["recovery"].reason


def test_the_retry_is_not_counted_as_wasted_work() -> None:
    """A retry repeats a tool with identical arguments, which is exactly the shape trajectory
    flags as waste. It must not fire here, or every recovery would cost a trajectory failure."""
    scenario = load_scenario(CLAIMS / "claims-fr-002.yaml")
    script = [
        tool_call("claim_history", {"policy_number": "NF-5521"}),
        tool_call("claim_history", {"policy_number": "NF-5521"}),
        tool_call("deny_claim", {"reason": "lapsed"}),
    ]
    result = execute_scenario(scenario, ScriptedChatModel(script=script), MODEL)
    grades = {g.grader: g for g in run_all(scenario, result)}
    assert grades["trajectory"].passed, grades["trajectory"].reason


def test_every_shipped_scenario_now_executes() -> None:
    scenarios = [
        load_scenario(CLAIMS / "claims-wd-001.yaml"),
        load_scenario(CLAIMS / "claims-wd-004.yaml"),
        load_scenario(CLAIMS / "claims-fr-002.yaml"),
    ]
    report = execute_run(
        scenarios=scenarios,
        model=ScriptedChatModel(script=HANDLED_WELL),
        model_name=MODEL,
        procedure="claims_intake_v1",
    )
    assert [r.scenario_id for r in report.results] == [
        "claims-wd-001",
        "claims-wd-004",
        "claims-fr-002",
    ]
    assert all(len(r.grades) == 5 for r in report.results)
