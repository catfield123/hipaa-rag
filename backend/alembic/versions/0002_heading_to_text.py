"""Relax document node heading column.

Revision ID: 0002_heading_to_text
Revises: 0001_initial
Create Date: 2026-04-10 00:30:00
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0002_heading_to_text"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "document_nodes",
        "heading",
        existing_type=sa.String(length=512),
        type_=sa.Text(),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "document_nodes",
        "heading",
        existing_type=sa.Text(),
        type_=sa.String(length=512),
        existing_nullable=True,
    )
