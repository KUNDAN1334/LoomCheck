"""Resolver tests: what the agent gets back, when it changes, and what it is refused."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loomcheck.config import PROJECT_ROOT
from loomcheck.loader import load_scenario
from loomcheck.mocks.failures import TRUNCATED_JSON
from loomcheck.mocks.resolver import MockResolver, UndeclaredToolError
from loomcheck.models import (
    CaseSpec,
    Expectation,
    GroundTruth,
    InjectedFailure,
    Outcome,
    Scenario,
    ToolMock,
)

CLAIMS = PROJECT_ROOT / "scenarios" / "claims"


def failing_scenario(failure: InjectedFailure, *, recovers: bool) -> Scenario:
    """A scenario whose only tool fails once, and either recovers or stays down."""
    return Scenario(
        id="t-001",
        description="a scenario",
        procedure="claims_intake_v1",
        case=CaseSpec(title="a case", inbound=Path("fixtures/emails/wd_001.txt")),
        ground_truth=GroundTruth(precedents=["PR-1"]),
        tools={
            "claim_history": ToolMock(
                failure=failure,
                then=Path("fixtures/tools/history_7710.json") if recovers else None,
            )
        },
        expect=Expectation(outcome=Outcome.DENY_CLAIM, max_tool_calls=6, max_cost_usd=0.5),
    )


def test_resolves_a_declared_tool_to_its_fixture() -> None:
    scenario = load_scenario(CLAIMS / "claims-wd-001.yaml")
    resolution = MockResolver(scenario).resolve("policy_lookup")
    assert "NF-7710" in resolution.payload
    assert resolution.injected_failure is None
    assert resolution.is_error is False


def test_the_same_tool_returns_the_same_thing_every_call() -> None:
    """A scenario fixes one response per tool, so repeated calls are not a hidden state machine
    — unless a failure is declared, which is the one thing the call count changes."""
    resolver = MockResolver(load_scenario(CLAIMS / "claims-wd-001.yaml"))
    assert resolver.resolve("policy_lookup") == resolver.resolve("policy_lookup")
    assert resolver.call_count("policy_lookup") == 2


def test_an_undeclared_tool_names_what_the_scenario_does_declare() -> None:
    resolver = MockResolver(load_scenario(CLAIMS / "claims-wd-001.yaml"))
    with pytest.raises(UndeclaredToolError) as exc:
        resolver.resolve("weather_lookup")
    assert "'weather_lookup'" in str(exc.value)
    assert "policy_lookup" in str(exc.value)


def test_the_first_call_fails_and_the_next_one_recovers() -> None:
    resolver = MockResolver(failing_scenario(InjectedFailure.SERVER_ERROR, recovers=True))

    first = resolver.resolve("claim_history")
    assert first.injected_failure is InjectedFailure.SERVER_ERROR
    assert first.is_error is True
    assert json.loads(first.payload)["status"] == 500

    second = resolver.resolve("claim_history")
    assert second.injected_failure is None
    assert "CL-2024-0912" in second.payload

    third = resolver.resolve("claim_history")
    assert third.payload == second.payload


def test_a_mock_with_no_then_is_a_permanent_outage() -> None:
    """Some tools are down for the whole case, and the right behaviour is to stop rather than
    retry forever. The scenario grammar allows saying so, so the resolver has to mean it."""
    resolver = MockResolver(failing_scenario(InjectedFailure.TIMEOUT, recovers=False))
    for _ in range(3):
        resolution = resolver.resolve("claim_history")
        assert resolution.injected_failure is InjectedFailure.TIMEOUT


def test_a_quiet_failure_does_not_flag_the_tool_channel() -> None:
    """malformed_json and empty_result 'succeed'. Flagging them would hand the agent a signal it
    would not have in production, and recovery would be grading an easier problem."""
    malformed = MockResolver(
        failing_scenario(InjectedFailure.MALFORMED_JSON, recovers=True)
    ).resolve("claim_history")
    assert malformed.is_error is False
    assert malformed.injected_failure is InjectedFailure.MALFORMED_JSON
    assert malformed.payload == TRUNCATED_JSON
    with pytest.raises(json.JSONDecodeError):
        json.loads(malformed.payload)

    empty = MockResolver(failing_scenario(InjectedFailure.EMPTY_RESULT, recovers=True)).resolve(
        "claim_history"
    )
    assert empty.is_error is False
    assert json.loads(empty.payload) == {}


def test_the_shipped_failure_scenarios_resolve() -> None:
    """claims-wd-004 and claims-fr-002 were skipped until this phase; they run now."""
    for name in ("claims-wd-004", "claims-fr-002"):
        resolver = MockResolver(load_scenario(CLAIMS / f"{name}.yaml"))
        assert resolver.resolve("claim_history").injected_failure is InjectedFailure.SERVER_ERROR
        assert resolver.resolve("claim_history").injected_failure is None
