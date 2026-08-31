"""Did the agent reach the right resolution?

A comparison, not an LLM judge. Expected outcomes are a closed set of four actions and the
agent signals its decision by calling a terminal tool, so the check is `==`. A judge would add
cost, latency and its own variance to a problem that does not have one, and injecting variance
into a regression harness defeats the harness.
"""

from __future__ import annotations

from loomcheck.graders.base import terminal_turn, verdict
from loomcheck.models import GradeResult, Scenario, ScenarioResult

NAME = "outcome"

QUOTE_LIMIT = 160
"""Long enough to see what the agent was doing instead, short enough that a failures list stays
scannable. The whole message is in `loomcheck show`."""


def _quote(message: str | None) -> str:
    if not message:
        return ""
    collapsed = " ".join(message.split())
    if len(collapsed) > QUOTE_LIMIT:
        collapsed = collapsed[: QUOTE_LIMIT - 1] + "…"
    return f'; it wrote: "{collapsed}"'


def grade_outcome(scenario: Scenario, result: ScenarioResult) -> GradeResult:
    expected = scenario.expect.outcome

    if result.outcome is None:
        # Quote what it wrote instead. "Never called a terminal tool" says the run failed; the
        # agent's own last words are what say why, and they are the only evidence there is —
        # prose produces no turn record.
        said = _quote(result.final_message)
        return verdict(
            NAME,
            False,
            f"expected {expected.value}, but the agent never called a terminal tool; "
            f"the run stopped after {len(result.turns)} turn(s){said}",
        )

    if result.outcome is expected:
        turn = terminal_turn(result)
        at = f" at turn {turn.index}" if turn else ""
        return verdict(NAME, True, f"reached {expected.value}{at}")

    turn = terminal_turn(result)
    at = f" at turn {turn.index}" if turn else ""
    return verdict(
        NAME,
        False,
        f"expected {expected.value}, agent called {result.outcome.value}{at}",
    )
