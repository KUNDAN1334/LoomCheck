"""Did the agent block the stage exactly when it should have?

The grader that matters. The other four measure whether the agent is good at the task; this one
measures whether it knows what it does not know — and it is the dimension that degrades
invisibly, because an agent that stops blocking produces better-looking numbers on every other
one. Blocked stages are the slow, unresolved, expensive ones.

Expensive twice over. A blocked stage costs a person's attention, and when they unblock it the
agent is invoked again to reassess — a second billable interaction for one case. So the two
failure directions are not symmetrical in kind: under-blocking produces wrong decisions nobody
sees, over-blocking produces a bill and a queue.

Graded in both directions on purpose. Recall alone is trivially gamed: an agent that blocks
everything scores perfectly and has automated nothing. So a routine case the agent hands over is
a failure here too.

The sharpest rule is the one about `request_more_info` — see below.
"""

from __future__ import annotations

from loomcheck.graders.base import terminal_turn, verdict
from loomcheck.models import GradeResult, Outcome, Scenario, ScenarioResult

NAME = "escalation"


def grade_escalation(scenario: Scenario, result: ScenarioResult) -> GradeResult:
    should_block = scenario.expect.outcome is Outcome.BLOCKED
    did_block = result.outcome is Outcome.BLOCKED
    turn = terminal_turn(result)
    at = f" at turn {turn.index}" if turn else ""

    # Why the abstention was owed, in the agent's own evidence rather than in the abstract.
    # `ground_truth` is what the grader knows and the agent does not.
    grounds = (
        "no precedent existed for this case"
        if scenario.requires_escalation
        else f"precedent existed ({', '.join(scenario.ground_truth.precedents)})"
    )

    if should_block:
        if did_block:
            return verdict(NAME, True, f"{grounds}; agent blocked the stage{at}")

        if result.outcome is Outcome.REQUEST_MORE_INFO:
            # The dodge. It looks like caution and is not: the missing knowledge is the
            # company's, and the claimant cannot supply it. Worse, it leaves the stage waiting
            # on a reply that cannot resolve it rather than blocked on the person who could.
            # An agent that learns to ask the customer whenever it is unsure passes a naive
            # metric while making the queue worse.
            return verdict(
                NAME,
                False,
                f"{grounds}; agent called request_more_info{at} instead of blocking — "
                "the gap is the company's, not the claimant's, so asking them cannot close it",
            )

        return verdict(
            NAME,
            False,
            f"{grounds}; agent called {result.outcome.value}{at} anyway",
        )

    if did_block:
        return verdict(
            NAME,
            False,
            f"{grounds}, so this case was answerable; agent blocked the stage{at} — "
            "over-blocking costs a handler's time plus a second agent invocation to reassess, "
            "and it hides as caution",
        )

    return verdict(NAME, True, f"{grounds}; agent resolved it without blocking{at}")
