"""record what the agent wrote when it ended without acting

Only a tool call produces a turn row, so an agent that writes prose and stops used to leave no
trace at all — and "never called a terminal tool" turned out to be the commonest real failure the
first time this suite met a live model. This column is the evidence that explains it.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-25
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("scenario_results", sa.Column("final_message", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("scenario_results", "final_message")
