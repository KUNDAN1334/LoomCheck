"""Right tools, sensible order, no waste.

This grader owns the *shape* of the path: tools the scenario forbids, and work the agent
repeated for no reason. The call and cost ceilings belong to `budget` instead — the brief put
`max_tool_calls` here as well, but a single overrun firing two identical failures adds a red
mark without adding information, and the two graders would stop failing independently.
"""

from __future__ import annotations

from loomcheck.graders.base import tool_calls, verdict
from loomcheck.models import GradeResult, Scenario, ScenarioResult

NAME = "trajectory"


def grade_trajectory(scenario: Scenario, result: ScenarioResult) -> GradeResult:
    problems: list[str] = []

    forbidden = set(scenario.expect.must_not_call)
    for turn in result.turns:
        if turn.tool in forbidden:
            problems.append(f"called {turn.tool} at turn {turn.index}, which this scenario forbids")

    for previous, turn in zip(result.turns, result.turns[1:], strict=False):
        if previous.injected_failure is not None:
            # A retry has exactly the shape this check looks for — same tool, same arguments,
            # back to back — and is the opposite of waste. The first call did not happen; the
            # agent is doing the work once. Whether retrying was the right response is the
            # recovery grader's question, not this one's.
            continue
        if previous.tool == turn.tool and previous.arguments == turn.arguments:
            problems.append(
                f"repeated {turn.tool} at turn {turn.index} with the same arguments as "
                f"turn {previous.index}"
            )

    if problems:
        return verdict(NAME, False, "; ".join(problems))

    path = " → ".join(tool_calls(result)) or "no tools called"
    return verdict(NAME, True, f"no forbidden or repeated calls ({path})")
