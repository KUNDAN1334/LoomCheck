"""Rendering tests.

There is exactly one of these because there was exactly one bug worth pinning: Rich reads
square brackets as style markup, so the list of allowed enum values in a loader error was
being swallowed and the message printed as "must be one of , got 'escalate'". The loader
was correct and the terminal lied about it, which is the worst kind of failure for a tool
whose whole output is terminal text.
"""

from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch
from rich.console import Console

from loomcheck import report
from loomcheck.loader import ScenarioError


def test_bracketed_error_text_survives_rich_markup(monkeypatch: MonkeyPatch) -> None:
    recorder = Console(record=True, width=200)
    monkeypatch.setattr(report, "console", recorder)

    report.render_load_errors(
        [
            ScenarioError(
                Path("scenarios/claims/claims-wd-004.yaml"),
                "'expect.outcome' must be one of [approve_claim, blocked], got 'escalate'",
                24,
            )
        ]
    )

    output = recorder.export_text()
    assert "[approve_claim, blocked]" in output
    assert "claims-wd-004.yaml:24" in output
