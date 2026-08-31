"""The vocabulary of the harness.

Everything the runner, graders and diff exchange is defined here so that a change to
the shape of a scenario is a change to exactly one file. The ORM tables in db.py mirror
these models; when one moves, the other has to move with it.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Outcome(StrEnum):
    """How the agent ends the stage.

    Closed on purpose: because it is an enum rather than free text, the outcome grader is a
    comparison instead of an LLM judge, which keeps the suite deterministic and free.

    Three of these are domain resolutions and one is a lifecycle state, which looks inconsistent
    until you map them onto what the stage does next:

        approve_claim, deny_claim   the stage completes
        request_more_info           the stage waits on external input — the claimant's reply
        blocked                     the stage halts and waits for a person

    `blocked` is named for the platform state rather than for the action because that is what it
    is: not "escalate", which is a description of intent, but the value the stage actually
    carries when it stops. A blocked stage resumes when a human posts a message or resumes it
    manually, and the agent is invoked again to reassess — so blocking costs a person's attention
    *and* a second invocation.

    Keeping `request_more_info` apart from `blocked` is what lets the escalation grader catch an
    agent that dodges a knowledge gap by asking the customer a question the customer cannot
    answer. Waiting on a claimant and halting for a colleague are different states with different
    costs, and collapsing them would hide the more expensive one.
    """

    APPROVE_CLAIM = "approve_claim"
    DENY_CLAIM = "deny_claim"
    REQUEST_MORE_INFO = "request_more_info"
    BLOCKED = "blocked"


class InjectedFailure(StrEnum):
    """Failure modes the mock resolver can inject in place of a tool response."""

    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"
    MALFORMED_JSON = "malformed_json"
    EMPTY_RESULT = "empty_result"


class ToolMock(BaseModel):
    """What the mock resolver should return when the agent calls one tool.

    Either a fixture is returned every time (`response`), or the first call fails with an
    injected `failure` and every later call returns `then`. A mock with a failure and no
    `then` is a permanent outage, which is a legitimate scenario, so `then` stays optional.
    """

    model_config = ConfigDict(extra="forbid")

    response: Path | None = None
    failure: InjectedFailure | None = None
    then: Path | None = None

    @model_validator(mode="after")
    def check_exclusive(self) -> ToolMock:
        if self.response is not None and self.failure is not None:
            raise ValueError("set either 'response' or 'failure', not both")
        if self.response is None and self.failure is None:
            raise ValueError("must set one of 'response' or 'failure'")
        if self.then is not None and self.failure is None:
            raise ValueError("'then' only makes sense alongside 'failure'")
        return self

    def fixture_paths(self) -> list[Path]:
        """Every fixture this mock will need at run time, for load-time existence checks."""
        return [p for p in (self.response, self.then) if p is not None]


class CaseSpec(BaseModel):
    """The inbound work item the agent is handed."""

    model_config = ConfigDict(extra="forbid")

    title: str
    inbound: Path


class GroundTruth(BaseModel):
    """Facts about the case that the graders may read and the agent may not.

    `precedents` is the list of precedent ids that genuinely exist for this case. An empty
    list is the signal that abstaining is the correct behaviour. The agent learns the same
    thing from the precedent_search mock; this field is how the escalation grader knows what
    the right answer was without re-deriving it from the trace.
    """

    model_config = ConfigDict(extra="forbid")

    precedents: list[str] = Field(default_factory=list)


class Expectation(BaseModel):
    """The bar this scenario has to clear."""

    model_config = ConfigDict(extra="forbid")

    outcome: Outcome
    must_not_call: list[str] = Field(default_factory=list)
    max_tool_calls: int = Field(gt=0)
    max_cost_usd: float = Field(gt=0)


class Scenario(BaseModel):
    """One YAML file, validated. The unit of work for the whole harness."""

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    procedure: str
    case: CaseSpec
    ground_truth: GroundTruth
    tools: dict[str, ToolMock]
    expect: Expectation

    @property
    def requires_escalation(self) -> bool:
        """True when no precedent exists, so the only honest ending is to block the stage."""
        return not self.ground_truth.precedents


class TurnRecord(BaseModel):
    """One step of the agent's execution, exactly as the recorder captured it."""

    model_config = ConfigDict(extra="forbid")

    index: int
    tool: str
    arguments: dict[str, Any]
    result: str
    injected_failure: InjectedFailure | None = None
    latency_ms: int
    tokens_in: int
    tokens_out: int
    cost_usd: float


class GradeResult(BaseModel):
    """One grader's verdict on one scenario run.

    `applicable` exists so that the recovery grader can report "no failure was injected here"
    without contributing a free pass to the aggregate. A suite whose pass rate is inflated by
    graders that had nothing to check is the exact dishonesty this project is about.

    `reason` must be specific enough to debug from without opening the trace.
    """

    model_config = ConfigDict(extra="forbid")

    grader: str
    applicable: bool = True
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    reason: str


class ScenarioResult(BaseModel):
    """What happened when one scenario was run once."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    repetition: int = 1
    outcome: Outcome | None = None
    turns: list[TurnRecord] = Field(default_factory=list)
    grades: list[GradeResult] = Field(default_factory=list)
    total_cost_usd: float = 0.0


class RunReport(BaseModel):
    """One invocation of `loomcheck run`, across every scenario it touched."""

    model_config = ConfigDict(extra="forbid")

    run_id: UUID
    started_at: datetime
    procedure: str
    model: str
    results: list[ScenarioResult] = Field(default_factory=list)

    def pass_rate(self, grader: str) -> float | None:
        """Share of applicable gradings this grader passed, or None if it never applied."""
        grades = [g for r in self.results for g in r.grades if g.grader == grader and g.applicable]
        if not grades:
            return None
        return sum(1 for g in grades if g.passed) / len(grades)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.total_cost_usd for r in self.results)
