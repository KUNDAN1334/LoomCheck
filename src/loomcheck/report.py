"""Rich terminal rendering.

The terminal is the whole interface, so output here is treated as a deliverable rather than as
logging: someone reading it should not need to open a scenario file to know what was checked,
or open the source to know what went wrong.

Failures print their full reason string. Truncating them would leave a red mark with no way to
act on it, which is most of the value gone.
"""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from loomcheck.db import RunSummary
from loomcheck.diff import RunDiff
from loomcheck.graders import GRADER_NAMES
from loomcheck.loader import ScenarioError
from loomcheck.models import RunReport

console = Console()


def render_load_errors(errors: list[ScenarioError]) -> None:
    """Print every bad file at once, so one editing pass fixes all of them.

    Messages are escaped because they contain square brackets — the list of allowed enum
    values, for one — and Rich would otherwise read them as style markup and print nothing
    where the allowed values should be.
    """
    for error in errors:
        location = f"{error.path}:{error.line}" if error.line is not None else str(error.path)
        console.print(f"[red]error[/red] {escape(location)}: {escape(error.message)}")


def _mark(report: RunReport, scenario_id: str, grader: str) -> str:
    grade = next(
        (
            g
            for r in report.results
            if r.scenario_id == scenario_id
            for g in r.grades
            if g.grader == grader
        ),
        None,
    )
    if grade is None:
        return "[dim]?[/dim]"
    if not grade.applicable:
        return "[dim]–[/dim]"
    return "[green]✓[/green]" if grade.passed else "[red]✗[/red]"


def render_run_report(report: RunReport) -> None:
    """One run: what each scenario did, how it graded, and why anything failed."""
    table = Table(
        title=f"run {str(report.run_id)[:8]}  ·  {report.procedure}  ·  {report.model}",
        title_justify="left",
    )
    table.add_column("scenario", style="bold", no_wrap=True)
    # "resolution", not "outcome": one of the grader columns is already called that, and two
    # columns under the same header is a table nobody can read out loud.
    table.add_column("resolution", no_wrap=True)
    for name in GRADER_NAMES:
        table.add_column(name, justify="center", no_wrap=True)
    table.add_column("turns", justify="right", no_wrap=True)
    table.add_column("cost", justify="right", no_wrap=True)

    for result in report.results:
        outcome = result.outcome.value if result.outcome else "[yellow]unresolved[/yellow]"
        marks = [_mark(report, result.scenario_id, name) for name in GRADER_NAMES]
        table.add_row(
            result.scenario_id,
            outcome,
            *marks,
            str(len(result.turns)),
            f"${result.total_cost_usd:.4f}",
        )

    console.print(table)
    _render_rates(report)

    failures = [
        (result.scenario_id, grade)
        for result in report.results
        for grade in result.grades
        if grade.applicable and not grade.passed
    ]
    if failures:
        console.print("\n[bold]failures[/bold]")
        for scenario_id, grade in failures:
            console.print(f"  [bold]{scenario_id}[/bold]  [red]{grade.grader}[/red]")
            console.print(f"    {escape(grade.reason)}")


def _render_rates(report: RunReport) -> None:
    """Pass rates, over applicable gradings only. A grader that had nothing to check reads
    `n/a` rather than contributing a free pass to the aggregate."""
    parts: list[str] = []
    for name in GRADER_NAMES:
        rate = report.pass_rate(name)
        parts.append(f"{name} n/a" if rate is None else f"{name} {rate:.0%}")
    console.print("  ".join(parts) + f"   ·   total ${report.total_cost_usd:.4f}")


def render_runs_table(summaries: list[RunSummary]) -> None:
    if not summaries:
        console.print("No runs yet. `loomcheck run scenarios/claims/` makes one.")
        return

    table = Table(title=f"{len(summaries)} run(s)", title_justify="left")
    table.add_column("id", style="bold", no_wrap=True)
    table.add_column("started", no_wrap=True)
    table.add_column("procedure", no_wrap=True)
    table.add_column("model", overflow="fold")
    table.add_column("scenarios", justify="right", no_wrap=True)
    table.add_column("cost", justify="right", no_wrap=True)

    for run in summaries:
        table.add_row(
            str(run.id)[:8],
            run.started_at.strftime("%Y-%m-%d %H:%M"),
            run.procedure,
            run.model,
            str(run.scenarios),
            f"${run.total_cost_usd:.4f}",
        )

    console.print(table)
    console.print("[dim]Ids are shown short; any unique prefix works.[/dim]")


def _brief(value: Any, limit: int = 70) -> str:
    text = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


def render_run_detail(report: RunReport) -> None:
    """One run, scenario by scenario, turn by turn, with every grader's reasoning.

    The trace is printed in full rather than summarised because a verdict is only worth
    trusting if the steps behind it can be read back.
    """
    console.print(
        f"run [bold]{report.run_id}[/bold]\n"
        f"{report.started_at:%Y-%m-%d %H:%M:%S %Z}  ·  {report.procedure}  ·  {report.model}\n"
    )

    for result in report.results:
        outcome = result.outcome.value if result.outcome else "unresolved"
        console.print(
            f"[bold]{result.scenario_id}[/bold]  →  {outcome}  "
            f"[dim]({len(result.turns)} turns, ${result.total_cost_usd:.4f})[/dim]"
        )
        turns = Table(show_header=True, box=None, pad_edge=False, padding=(0, 2))
        turns.add_column("#", justify="right", no_wrap=True)
        turns.add_column("tool", no_wrap=True)
        turns.add_column("arguments", overflow="fold")
        turns.add_column("result", overflow="fold")
        turns.add_column("ms", justify="right", no_wrap=True)
        turns.add_column("tok", justify="right", no_wrap=True)

        for turn in result.turns:
            failure = f" [yellow]!{turn.injected_failure}[/yellow]" if turn.injected_failure else ""
            turns.add_row(
                str(turn.index),
                escape(turn.tool) + failure,
                escape(_brief(turn.arguments)),
                escape(_brief(turn.result)),
                str(turn.latency_ms),
                f"{turn.tokens_in}/{turn.tokens_out}",
            )
        console.print(turns)

        if result.final_message:
            # The run ended without acting. This is the only record of what it did instead.
            console.print("\n  [yellow]ended without acting. it wrote:[/yellow]")
            console.print(f"  [dim]{escape(result.final_message.strip())}[/dim]")

        for grade in result.grades:
            if not grade.applicable:
                symbol, colour = "–", "dim"
            elif grade.passed:
                symbol, colour = "✓", "green"
            else:
                symbol, colour = "✗", "red"
            console.print(
                f"  [{colour}]{symbol} {grade.grader:<11}[/{colour}] {escape(grade.reason)}"
            )
        console.print()


def _rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def render_diff(diff: RunDiff) -> None:
    """The comparison the whole project exists to print.

    Deliberately not a boxed table. A diff is read down the delta column, and box borders put
    furniture between the two numbers a reader is trying to subtract.
    """
    arrow = f"{diff.before.procedure} → {diff.after.procedure}"
    console.print(
        f"[bold]{escape(arrow)}[/bold]"
        if diff.procedure_changed
        else f"[bold]{escape(diff.before.procedure)}[/bold] (unchanged)"
    )
    model = (
        f"{diff.before.model} → {diff.after.model}"
        if diff.model_changed
        else f"{diff.before.model} (unchanged)"
    )
    console.print(f"[dim]model {escape(model)}[/dim]")
    console.print(
        f"[dim]run {str(diff.before.run_id)[:8]} → {str(diff.after.run_id)[:8]}"
        f"  ·  {len(diff.shared_scenarios)} scenario(s) compared[/dim]\n"
    )

    if diff.confounded:
        # Worth shouting about. With both variables moved there is no honest attribution.
        console.print(
            "[yellow]⚠ the model changed as well as the procedure — this comparison cannot "
            "attribute any change to either[/yellow]\n"
        )

    if diff.only_before or diff.only_after:
        console.print(
            "[yellow]⚠ the two runs cover different scenarios; rates below are over the "
            f"{len(diff.shared_scenarios)} they share, so they will not match "
            "`loomcheck show`[/yellow]"
        )
        if diff.only_before:
            console.print(f"[dim]  only in the earlier run: {', '.join(diff.only_before)}[/dim]")
        if diff.only_after:
            console.print(f"[dim]  only in the later run:   {', '.join(diff.only_after)}[/dim]")
        console.print()

    table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 1))
    table.add_column("metric", no_wrap=True, min_width=12)
    table.add_column("before", justify="right", no_wrap=True, min_width=8)
    table.add_column("arrow", no_wrap=True)
    table.add_column("after", justify="right", no_wrap=True, min_width=8)
    table.add_column("delta", justify="right", no_wrap=True, min_width=6)
    table.add_column("flag", no_wrap=True)

    for metric in diff.metrics:
        if metric.delta is None:
            note = "newly covered" if metric.after is not None else "no longer covered"
            movement, flag = "·", f"[dim]{note}[/dim]"
        else:
            # A metric that did not move gets a blank rather than "+0%": the delta column is
            # read by scanning for signs, and a column of zeros is noise to scan past.
            movement = f"{metric.delta:+.0%}" if metric.delta else ""
            flag = "[red]⚠ REGRESSION[/red]" if metric.is_regression else ""
        table.add_row(metric.name, _rate(metric.before), "→", _rate(metric.after), movement, flag)

    ratio = (diff.cost_after - diff.cost_before) / diff.cost_before if diff.cost_before else 0.0
    cost_change = f"{ratio:+.0%}" if round(ratio, 2) else ""
    # Cost is shown but never flagged as a regression. Spend falling is the thing that happens
    # when an agent stops doing work, so calling it good or bad here would prejudge the case
    # the reader is being asked to look at.
    table.add_row(
        "cost / case", f"${diff.cost_before:.4f}", "→", f"${diff.cost_after:.4f}", cost_change, ""
    )
    console.print(table)

    if not diff.changes:
        console.print("\nNo scenario changed verdict.")
    else:
        console.print(f"\n[bold]{len(diff.changes)} scenario(s) changed verdict[/bold]")
        for change in diff.changes:
            before = change.before_outcome.value if change.before_outcome else "unresolved"
            after = change.after_outcome.value if change.after_outcome else "unresolved"
            heading = f"  [bold]{change.scenario_id}[/bold]"
            if change.outcome_changed:
                heading += f"   {escape(before)} → {escape(after)}"
            console.print(heading)
            for flip in change.flips:
                mark = "[red]✗[/red]" if flip.was_passing else "[green]✓[/green]"
                console.print(f"    {mark} {flip.grader}")
                console.print(f"      [dim]{escape(flip.reason)}[/dim]")

    _render_closing(diff)


def _render_closing(diff: RunDiff) -> None:
    """The sentence the tool exists for — printed only when it is true.

    A canned line would be worse than none: it has to name the numbers that actually moved,
    or it is the same unearned confidence the harness is built to catch.
    """
    if not diff.dashboard_would_be_green:
        return

    better = [m.name for m in diff.improvements]
    if diff.cost_after < diff.cost_before:
        better.append("cost per case")
    worse = [m.name for m in diff.regressions]
    headline = f"{_join(better).capitalize()} improved. {_join(worse).capitalize()} did not."
    console.print(f"\n[bold]{headline}[/bold]\nA dashboard would have shown green.")


def _join(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"
