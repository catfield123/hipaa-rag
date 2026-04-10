"""Initial HIPAA RAG schema.

Revision ID: 0001_initial
Revises:
Create Date: 2026-04-10 00:00:00
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql


revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_textsearch")

    op.create_table(
        "retrieval_chunks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=False),
        sa.Column("path", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("path_text", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column(
            "search_text",
            sa.Text(),
            sa.Computed("coalesce(path_text, '') || E'\\n' || coalesce(text, '')", persisted=True),
            nullable=False,
        ),
        sa.Column("section", sa.String(length=255)),
        sa.Column("part", sa.String(length=255)),
        sa.Column("subpart", sa.String(length=255)),
        sa.Column("markers", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("embedding", Vector(1536)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_retrieval_chunks_section", "retrieval_chunks", ["section"])
    op.create_index("ix_retrieval_chunks_part", "retrieval_chunks", ["part"])
    op.create_index("ix_retrieval_chunks_subpart", "retrieval_chunks", ["subpart"])
    op.execute(
        """
        CREATE INDEX retrieval_chunks_search_text_bm25_idx
        ON retrieval_chunks
        USING bm25 (search_text)
        WITH (text_config = 'english')
        """
    )
    op.execute(
        """
        CREATE INDEX retrieval_chunks_embedding_hnsw_idx
        ON retrieval_chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS retrieval_chunks_embedding_hnsw_idx")
    op.execute("DROP INDEX IF EXISTS retrieval_chunks_search_text_bm25_idx")
    op.drop_index("ix_retrieval_chunks_subpart", table_name="retrieval_chunks")
    op.drop_index("ix_retrieval_chunks_part", table_name="retrieval_chunks")
    op.drop_index("ix_retrieval_chunks_section", table_name="retrieval_chunks")
    op.drop_table("retrieval_chunks")
