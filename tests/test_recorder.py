"""Recorder tests: what counts as a turn, and where cost goes when nothing does."""

from __future__ import annotations

from loomcheck.models import Outcome
from loomcheck.recorder import Recorder

MODEL = "llama-3.3-70b-versatile"


def test_a_turn_pairs_the_model_call_with_the_tool_call_it_produced() -> None:
    recorder = Recorder("t-001", MODEL)
    recorder.observe_model_call(latency_ms=420, tokens_in=1000, tokens_out=40)
    recorder.capture("policy_lookup", {"policy_number": "NF-7710"}, "{}")

    result = recorder.finish(Outcome.APPROVE_CLAIM)
    assert len(result.turns) == 1
    turn = result.turns[0]
    assert (turn.index, turn.tool, turn.latency_ms) == (1, "policy_lookup", 420)
    assert turn.tokens_in == 1000
    assert turn.cost_usd > 0


def test_turns_are_numbered_from_one_so_a_reason_string_can_cite_them() -> None:
    recorder = Recorder("t-001", MODEL)
    for tool in ("policy_lookup", "precedent_search", "blocked"):
        recorder.observe_model_call(latency_ms=1, tokens_in=10, tokens_out=1)
        recorder.capture(tool, {}, "{}")
    assert [t.index for t in recorder.finish(Outcome.BLOCKED).turns] == [1, 2, 3]


def test_a_model_call_that_produced_no_tool_call_still_costs_money() -> None:
    """The agent writing prose instead of acting is spend with nothing to attribute it to.
    Dropping it would let an agent talk its way past a cost ceiling for free."""
    recorder = Recorder("t-001", MODEL)
    recorder.observe_model_call(latency_ms=100, tokens_in=1000, tokens_out=40)  # prose
    recorder.observe_model_call(latency_ms=100, tokens_in=1000, tokens_out=40)  # then acts
    recorder.capture("policy_lookup", {}, "{}")

    result = recorder.finish(None)
    assert len(result.turns) == 1
    assert result.total_cost_usd > result.turns[0].cost_usd


def test_an_unresolved_run_records_no_outcome_rather_than_a_wrong_one() -> None:
    recorder = Recorder("t-001", MODEL)
    recorder.observe_model_call(latency_ms=1, tokens_in=10, tokens_out=1)
    assert recorder.finish(None).outcome is None
