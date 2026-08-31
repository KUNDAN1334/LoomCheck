"""The interface. Four commands, no more."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from sqlalchemy.orm import Session, sessionmaker

from loomcheck.config import ConfigError, load_database_settings, load_llm_settings
from loomcheck.db import (
    RunNotFoundError,
    list_runs,
    load_run,
    make_engine,
    make_session_factory,
    resolve_run_id,
    save_run,
)
from loomcheck.diff import compare
from loomcheck.loader import (
    ScenarioError,
    ScenarioValidationError,
    load_scenarios,
    procedure_path,
)
from loomcheck.report import (
    console,
    render_diff,
    render_load_errors,
    render_run_detail,
    render_run_report,
    render_runs_table,
)
from loomcheck.runner import execute_run, make_model, preflight

app = typer.Typer(add_completion=False)


@app.callback()
def main() -> None:
    """Regression and drift harness for multi-step AI agent workflows.

    Typer folds a single-command app into a bare executable, which would make the CLI
    `loomcheck <path>` today and `loomcheck run <path>` once a second command exists.
    Declaring the group here keeps the invocation stable as commands are added.
    """


def _fail(message: str) -> typer.Exit:
    console.print(f"[red]error[/red] {message}")
    return typer.Exit(1)


def _session_factory() -> sessionmaker[Session]:
    engine = make_engine(load_database_settings().url)
    # Connect before anything expensive happens: discovering the database is down after
    # spending a suite's worth of tokens is the wrong order to find out.
    engine.connect().close()
    return make_session_factory(engine)


@app.command()
def run(
    target: Annotated[
        Path,
        typer.Argument(help="A scenario .yaml file, or a directory of them."),
    ],
    procedure: Annotated[
        str | None,
        typer.Option(
            "--procedure",
            help="Run against this procedure instead of the one the scenarios name.",
        ),
    ] = None,
) -> None:
    """Run scenarios against the agent, save the trace, and report.

    `--procedure` is how a procedure edit gets tested: run the same suite against v1 and v2 and
    diff the two runs. Without it, comparing versions would mean editing every scenario file and
    editing it back, which is the sort of chore that stops a check from being run at all.
    """
    try:
        scenarios = load_scenarios(target)
        if procedure is not None:
            procedure_path(procedure)  # fail on a typo now, not on the first scenario
    except ScenarioValidationError as exc:
        render_load_errors(exc.errors)
        raise typer.Exit(1) from exc
    except ScenarioError as exc:
        render_load_errors([exc])
        raise typer.Exit(1) from exc

    try:
        sessions = _session_factory()
        llm = load_llm_settings()
        preflight(llm)
    except ConfigError as exc:
        raise _fail(str(exc)) from exc

    report = execute_run(
        scenarios=scenarios,
        model=make_model(llm),
        model_name=llm.model,
        procedure=procedure or scenarios[0].procedure,
    )

    with sessions() as session:
        save_run(session, report)

    render_run_report(report)


@app.command("list")
def list_command(
    limit: Annotated[int, typer.Option(help="How many runs to show.")] = 20,
) -> None:
    """List past runs, newest first."""
    try:
        sessions = _session_factory()
    except ConfigError as exc:
        raise _fail(str(exc)) from exc

    with sessions() as session:
        render_runs_table(list_runs(session, limit))


@app.command()
def show(
    run_id: Annotated[str, typer.Argument(help="A run id, or any unique prefix of one.")],
) -> None:
    """Show one run in full: every scenario, every turn."""
    try:
        sessions = _session_factory()
    except ConfigError as exc:
        raise _fail(str(exc)) from exc

    with sessions() as session:
        try:
            render_run_detail(load_run(session, resolve_run_id(session, run_id)))
        except RunNotFoundError as exc:
            raise _fail(str(exc)) from exc


@app.command()
def diff(
    run_a: Annotated[str, typer.Argument(help="The earlier run id, or a unique prefix.")],
    run_b: Annotated[str, typer.Argument(help="The later run id, or a unique prefix.")],
) -> None:
    """Compare two runs and show what regressed."""
    try:
        sessions = _session_factory()
    except ConfigError as exc:
        raise _fail(str(exc)) from exc

    with sessions() as session:
        try:
            before = load_run(session, resolve_run_id(session, run_a))
            after = load_run(session, resolve_run_id(session, run_b))
        except RunNotFoundError as exc:
            raise _fail(str(exc)) from exc

    render_diff(compare(before, after))
