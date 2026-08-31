"""Persistence roundtrip.

Skipped when DATABASE_URL is unset, and pytest is configured with `-ra` so the skip is
printed rather than passing by silently. A skipped test that looks like a green suite is the
same lie this project is built to catch.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from loomcheck.db import (
    Base,
    Run,
    RunNotFoundError,
    list_runs,
    load_run,
    make_engine,
    make_session_factory,
    resolve_run_id,
    save_run,
)
from loomcheck.models import GradeResult, Outcome, RunReport, ScenarioResult, TurnRecord

DATABASE_URL = os.environ.get("DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL, reason="DATABASE_URL not set; start Postgres with `docker compose up -d db`"
)


def _report() -> RunReport:
    return RunReport(
        run_id=uuid4(),
        started_at=datetime.now(UTC),
        procedure="claims_intake_v1",
        model="llama-3.3-70b-versatile",
        results=[
            ScenarioResult(
                scenario_id="claims-wd-004",
                outcome=Outcome.BLOCKED,
                total_cost_usd=0.0031,
                turns=[
                    TurnRecord(
                        index=1,
                        tool="precedent_search",
                        arguments={"query": "mixed use, commercial appliance"},
                        result='{"results": []}',
                        latency_ms=412,
                        tokens_in=1800,
                        tokens_out=60,
                        cost_usd=0.0011,
                    )
                ],
                grades=[
                    GradeResult(
                        grader="escalation",
                        passed=True,
                        score=1.0,
                        reason=("precedent_search returned 0 results; agent blocked at turn 2"),
                    )
                ],
            )
        ],
    )


@pytest.fixture
def sessions() -> Iterator[sessionmaker[Session]]:
    """Hand out sessions, then remove whatever the test created.

    These run against the developer's own database, and a test suite that leaves rows behind
    would fill `loomcheck list` with runs nobody made.
    """
    engine = make_engine(DATABASE_URL or "")
    Base.metadata.create_all(engine, checkfirst=True)
    factory = make_session_factory(engine)

    with factory() as session:
        existing = {run.id for run in session.scalars(select(Run)).all()}

    yield factory

    with factory() as session:
        for run in session.scalars(select(Run)).all():
            if run.id not in existing:
                session.delete(run)
        session.commit()


def test_a_run_survives_the_roundtrip_unchanged(sessions: sessionmaker[Session]) -> None:
    """`show` and `diff` read runs back, so what comes out has to be what went in."""
    report = _report()
    with sessions() as session:
        save_run(session, report)
        restored = load_run(session, report.run_id)

    assert restored.procedure == report.procedure
    assert restored.model == report.model
    result = restored.results[0]
    assert result.outcome is Outcome.BLOCKED
    assert result.repetition == 1
    assert result.turns[0].arguments == {"query": "mixed use, commercial appliance"}
    assert result.grades[0].reason.startswith("precedent_search returned 0 results")
    assert restored.pass_rate("escalation") == 1.0


def test_a_saved_run_appears_in_the_listing(sessions: sessionmaker[Session]) -> None:
    report = _report()
    with sessions() as session:
        save_run(session, report)
        assert report.run_id in {run.id for run in list_runs(session, limit=50)}


def test_a_run_can_be_addressed_by_a_prefix(sessions: sessionmaker[Session]) -> None:
    report = _report()
    with sessions() as session:
        save_run(session, report)
        assert resolve_run_id(session, str(report.run_id)[:8]) == report.run_id
        with pytest.raises(RunNotFoundError, match="no run starts with"):
            resolve_run_id(session, "zzzzzzzz")
