"""Orchestration: a plain loop over scenarios.

No workflow engine. The work is a bounded, in-process pass over a dozen scenarios that
finishes in minutes; durability across restarts buys nothing when re-running is cheap.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from groq import Groq, GroqError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from pydantic import SecretStr

from loomcheck.agent.graph import build_agent, initial_state
from loomcheck.agent.tools import tool_schemas
from loomcheck.config import PRICES, PROJECT_ROOT, ConfigError, LLMSettings
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


def preflight(settings: LLMSettings) -> None:
    """Refuse to start a run the provider or the price table cannot support.

    Same family as the database connection check in the CLI: anything knowably broken before
    tokens are spent should be found before tokens are spent. This one exists because a model an
    account cannot reach used to surface as a sixty-line HTTP traceback, on the first scenario,
    after the suite had already started — the opposite of how every other failure here reads.

    Three checks, in the order that produces the most useful message soonest:

    1. Is the model on the account at all? Its message can list what the key *can* use, which is
       what the reader needs next.
    2. Does it have a price? An unpriced model would report a suite that cost nothing.
    3. Will it actually take a tool call? Half the models on a Groq account are transcription,
       speech or classifier models that answer chat and refuse tools, and some need terms
       accepted in the console first. Neither shows up in the model listing, so the only honest
       test is to make one tiny call with the real tool schemas bound and see what comes back.
       It costs about a hundred tokens and it is the difference between failing now and failing
       on scenario one.
    """
    try:
        listing = Groq(api_key=settings.api_key).models.list()
    except GroqError as exc:
        raise ConfigError(f"could not ask Groq which models this key can use: {exc}") from exc

    available = sorted(model.id for model in listing.data)
    if settings.model not in available:
        raise ConfigError(
            f"LOOMCHECK_MODEL is {settings.model!r}, which this API key cannot use. "
            f"Available to it: {', '.join(available)}. "
            "Set LOOMCHECK_MODEL in .env to one of those."
        )

    if settings.model not in PRICES:
        known = ", ".join(sorted(PRICES))
        raise ConfigError(
            f"{settings.model!r} is available but has no price recorded, so cost could not be "
            f"reported honestly. Add it to PRICES in config.py (priced already: {known})."
        )

    try:
        make_model(settings).bind_tools(tool_schemas(), parallel_tool_calls=False).invoke(
            [HumanMessage(content="Reply with the single word: ready.")]
        )
    except GroqError as exc:
        raise ConfigError(
            f"{settings.model!r} is on this account but will not serve the request this harness "
            f"makes. Groq said: {exc}. Every scenario needs a chat model that accepts tool "
            "calling, so transcription, speech and classifier models will not work here."
        ) from exc


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
