"""Persistence.

Runs live in Postgres rather than in JSON files because the whole point of the tool is
comparing a run to one from last week, and answering "which scenarios changed verdict"
across two runs is a join, not a directory walk.

These tables mirror the Pydantic models in models.py. A change to one needs a change to
the other plus an Alembic revision.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from loomcheck.models import (
    GradeResult,
    InjectedFailure,
    Outcome,
    RunReport,
    ScenarioResult,
    TurnRecord,
)


class Base(DeclarativeBase):
    pass


class Run(Base):
    """One invocation of `loomcheck run`. The unit a diff compares."""

    __tablename__ = "runs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    procedure: Mapped[str] = mapped_column(String(128))
    model: Mapped[str] = mapped_column(String(128))

    results: Mapped[list[ScenarioResultRow]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class ScenarioResultRow(Base):
    """One scenario, run once, inside one run.

    `repetition` is 1 for every row today: v0 runs each scenario once. The column exists now
    because the honest answer to "is this a regression or model variance" is n>1, and adding
    the column later would mean migrating rows that already exist.
    """

    __tablename__ = "scenario_results"
    __table_args__ = (UniqueConstraint("run_id", "scenario_id", "repetition"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    scenario_id: Mapped[str] = mapped_column(String(128), index=True)
    repetition: Mapped[int] = mapped_column(Integer, default=1)
    outcome: Mapped[str | None] = mapped_column(String(32))
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    final_message: Mapped[str | None] = mapped_column(Text)
    """What the agent wrote instead of acting. Only set on an unresolved run, and the only
    evidence there is for the commonest real failure — see models.ScenarioResult."""

    run: Mapped[Run] = relationship(back_populates="results")
    turns: Mapped[list[TurnRow]] = relationship(
        back_populates="result_row", cascade="all, delete-orphan"
    )
    grades: Mapped[list[GradeRow]] = relationship(
        back_populates="result", cascade="all, delete-orphan"
    )


class TurnRow(Base):
    """One recorded step. Stored in full because a grader's verdict is only trustworthy
    if the trace behind it can be read back."""

    __tablename__ = "turns"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    scenario_result_id: Mapped[UUID] = mapped_column(
        ForeignKey("scenario_results.id", ondelete="CASCADE"), index=True
    )
    index: Mapped[int] = mapped_column("turn_index", Integer)
    tool: Mapped[str] = mapped_column(String(64))
    arguments: Mapped[dict[str, Any]] = mapped_column(JSONB)
    result: Mapped[str] = mapped_column(Text)
    injected_failure: Mapped[str | None] = mapped_column(String(32))
    latency_ms: Mapped[int] = mapped_column(Integer)
    tokens_in: Mapped[int] = mapped_column(Integer)
    tokens_out: Mapped[int] = mapped_column(Integer)
    cost_usd: Mapped[float] = mapped_column(Float)

    result_row: Mapped[ScenarioResultRow] = relationship(back_populates="turns")


class GradeRow(Base):
    """One grader's verdict, with the reason string kept verbatim: the diff prints it."""

    __tablename__ = "grades"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    scenario_result_id: Mapped[UUID] = mapped_column(
        ForeignKey("scenario_results.id", ondelete="CASCADE"), index=True
    )
    grader: Mapped[str] = mapped_column(String(32), index=True)
    applicable: Mapped[bool] = mapped_column(Boolean, default=True)
    passed: Mapped[bool] = mapped_column(Boolean)
    score: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text)

    result: Mapped[ScenarioResultRow] = relationship(back_populates="grades")


def make_engine(database_url: str) -> Engine:
    return create_engine(database_url, future=True)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, future=True)


@dataclass(frozen=True)
class RunSummary:
    """One row of `loomcheck list`."""

    id: UUID
    started_at: datetime
    procedure: str
    model: str
    scenarios: int
    total_cost_usd: float


class RunNotFoundError(Exception):
    """No run matches the id or prefix given."""


def save_run(session: Session, report: RunReport) -> None:
    """Write a whole run in one transaction. A half-saved run is not a run."""
    run = Run(
        id=report.run_id,
        started_at=report.started_at,
        procedure=report.procedure,
        model=report.model,
    )
    for result in report.results:
        row = ScenarioResultRow(
            run_id=report.run_id,
            scenario_id=result.scenario_id,
            repetition=result.repetition,
            outcome=result.outcome.value if result.outcome else None,
            total_cost_usd=result.total_cost_usd,
            final_message=result.final_message,
        )
        row.turns = [
            TurnRow(
                index=turn.index,
                tool=turn.tool,
                arguments=turn.arguments,
                result=turn.result,
                injected_failure=turn.injected_failure.value if turn.injected_failure else None,
                latency_ms=turn.latency_ms,
                tokens_in=turn.tokens_in,
                tokens_out=turn.tokens_out,
                cost_usd=turn.cost_usd,
            )
            for turn in result.turns
        ]
        row.grades = [
            GradeRow(
                grader=grade.grader,
                applicable=grade.applicable,
                passed=grade.passed,
                score=grade.score,
                reason=grade.reason,
            )
            for grade in result.grades
        ]
        run.results.append(row)

    session.add(run)
    session.commit()


def list_runs(session: Session, limit: int = 20) -> list[RunSummary]:
    runs = session.scalars(select(Run).order_by(Run.started_at.desc()).limit(limit)).all()
    return [
        RunSummary(
            id=run.id,
            started_at=run.started_at,
            procedure=run.procedure,
            model=run.model,
            scenarios=len(run.results),
            total_cost_usd=sum(r.total_cost_usd for r in run.results),
        )
        for run in runs
    ]


def resolve_run_id(session: Session, prefix: str) -> UUID:
    """Accept a unique id prefix, because nobody retypes a UUID to diff two runs."""
    matches = [
        run.id
        for run in session.scalars(select(Run)).all()
        if str(run.id).startswith(prefix.lower())
    ]
    if not matches:
        raise RunNotFoundError(f"no run starts with {prefix!r}; try `loomcheck list`")
    if len(matches) > 1:
        found = ", ".join(str(m)[:12] for m in sorted(matches, key=str))
        raise RunNotFoundError(f"{prefix!r} matches {len(matches)} runs: {found}")
    return matches[0]


def load_run(session: Session, run_id: UUID) -> RunReport:
    """Rebuild a RunReport from its rows, so `show` and `diff` read the same shape a run
    produced rather than a second, parallel representation of it."""
    run = session.get(Run, run_id)
    if run is None:
        raise RunNotFoundError(f"no run with id {run_id}")
    return RunReport(
        run_id=run.id,
        started_at=run.started_at,
        procedure=run.procedure,
        model=run.model,
        results=[
            ScenarioResult(
                scenario_id=row.scenario_id,
                repetition=row.repetition,
                outcome=Outcome(row.outcome) if row.outcome else None,
                total_cost_usd=row.total_cost_usd,
                final_message=row.final_message,
                turns=[
                    TurnRecord(
                        index=turn.index,
                        tool=turn.tool,
                        arguments=turn.arguments,
                        result=turn.result,
                        injected_failure=(
                            InjectedFailure(turn.injected_failure)
                            if turn.injected_failure
                            else None
                        ),
                        latency_ms=turn.latency_ms,
                        tokens_in=turn.tokens_in,
                        tokens_out=turn.tokens_out,
                        cost_usd=turn.cost_usd,
                    )
                    for turn in sorted(row.turns, key=lambda t: t.index)
                ],
                grades=[
                    GradeResult(
                        grader=grade.grader,
                        applicable=grade.applicable,
                        passed=grade.passed,
                        score=grade.score,
                        reason=grade.reason,
                    )
                    for grade in row.grades
                ],
            )
            for row in run.results
        ],
    )
