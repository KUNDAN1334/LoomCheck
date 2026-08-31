"""Orchestration: a plain loop over scenarios.

No workflow engine. The work is a bounded, in-process pass over a dozen scenarios that
finishes in minutes; durability across restarts buys nothing when re-running is cheap.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from pydantic import SecretStr

from loomcheck.agent.graph import build_agent, initial_state
from loomcheck.config import PROJECT_ROOT, LLMSettings
from loomcheck.graders import run_all
from loomcheck.loader import procedure_path
from loomcheck.mocks.resolver import MockResolver
from loomcheck.models import RunReport, Scenario, ScenarioResult
from loomcheck.recorder import Recorder

MAX_TURNS = 12
"""An agent that never calls a terminal tool has to stop somewhere. Not a config value: no
scenario has a reason to want a different ceiling, and the outcome grader already treats an
unresolved run as a failure rather than a crash."""


def make_model(settings: LLMSettings) -> BaseChatModel:
    """The only place a provider is named. See docs/learning.md D-002.

    Temperature 0 removes the variance that can be removed. The client's own transport
    retries are left at their default: a retried HTTP request is not a retried *tool* call,
    and only the latter is what the recovery grader is about.
    """
    return ChatGroq(
        model_name=settings.model,
        temperature=0.0,
        groq_api_key=SecretStr(settings.api_key),
    )


def execute_scenario(
    scenario: Scenario,
    model: BaseChatModel,
    model_name: str,
    root: Path = PROJECT_ROOT,
    procedure: str | None = None,
) -> ScenarioResult:
    """Run one scenario to a resolution and return its recorded trace.

    Builds a resolver and a recorder for this scenario alone, assembles the agent around
    them, and drives it until a terminal tool is called or MAX_TURNS is reached.

    `procedure` overrides the one the scenario names. That override is the whole mechanism
    behind the diff: running the same suite against v1 and v2 is how a procedure edit gets
    attributed to a behaviour change, and it would be unusable if it meant editing every
    scenario file twice.
    """
    resolver = MockResolver(scenario, root)
    recorder = Recorder(scenario.id, model_name)
    agent = build_agent(model, resolver, recorder, MAX_TURNS)

    instructions = procedure_path(procedure or scenario.procedure, root).read_text(encoding="utf-8")
    inbound = (root / scenario.case.inbound).read_text(encoding="utf-8")
    case = f"{scenario.case.title}\n\n{inbound}"

    final = agent.invoke(
        initial_state(instructions, case),
        {"recursion_limit": MAX_TURNS * 2 + 10},
    )
    return recorder.finish(final["outcome"])


def execute_run(
    scenarios: list[Scenario],
    model: BaseChatModel,
    model_name: str,
    procedure: str,
    root: Path = PROJECT_ROOT,
) -> RunReport:
    """Run every scenario in order and grade each trace as it comes back."""
    report = RunReport(
        run_id=uuid4(),
        started_at=datetime.now(UTC),
        procedure=procedure,
        model=model_name,
    )

    for scenario in scenarios:
        result = execute_scenario(scenario, model, model_name, root, procedure)
        # Graded here rather than inside execute_scenario, which has one job: produce a trace.
        # Grading a trace needs no model and no network, so it can be re-run over stored runs.
        result.grades = run_all(scenario, result)
        report.results.append(result)

    return report
