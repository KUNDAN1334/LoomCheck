"""The five graders.

Five and not one, because a single pass/fail cannot say *what* got worse, and what got worse
is the entire output of this tool. They fail independently: an agent can reach the right
outcome by a wasteful route, or the wrong outcome cheaply, or the right outcome only because
it ignored a tool failure and got lucky on stale data.

`GRADERS` is a tuple of five function references, not a registry. There is nothing to register
with and nothing to look up by name.
"""

from __future__ import annotations

from loomcheck.graders.base import Grader
from loomcheck.graders.budget import NAME as BUDGET
from loomcheck.graders.budget import grade_budget
from loomcheck.graders.escalation import NAME as ESCALATION
from loomcheck.graders.escalation import grade_escalation
from loomcheck.graders.outcome import NAME as OUTCOME
from loomcheck.graders.outcome import grade_outcome
from loomcheck.graders.recovery import NAME as RECOVERY
from loomcheck.graders.recovery import grade_recovery
from loomcheck.graders.trajectory import NAME as TRAJECTORY
from loomcheck.graders.trajectory import grade_trajectory
from loomcheck.models import GradeResult, Scenario, ScenarioResult

GRADERS: tuple[Grader, ...] = (
    grade_outcome,
    grade_trajectory,
    grade_escalation,
    grade_recovery,
    grade_budget,
)

GRADER_NAMES: tuple[str, ...] = (OUTCOME, TRAJECTORY, ESCALATION, RECOVERY, BUDGET)
"""The order graders are reported in, taken from the graders themselves. Lives here rather than
in `report.py` because the diff needs it too, and a rendering module is the wrong thing for a
computation module to depend on."""


def run_all(scenario: Scenario, result: ScenarioResult) -> list[GradeResult]:
    """Grade one recorded trace on every dimension. No grader calls a model."""
    return [grade(scenario, result) for grade in GRADERS]


__all__ = ["GRADERS", "GRADER_NAMES", "Grader", "run_all"]
