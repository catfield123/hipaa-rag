"""Initial HIPAA RAG schema.

Revision ID: 0001_initial
Revises:
Create Date: 2026-04-10 00:00:00
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


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
        sa.Column("path", sa.JSON(), nullable=False),
        sa.Column("path_text", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("section", sa.String(length=255)),
        sa.Column("part", sa.String(length=255)),
        sa.Column("subpart", sa.String(length=255)),
        sa.Column("markers", sa.JSON(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("embedding", Vector(1536)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_retrieval_chunks_section", "retrieval_chunks", ["section"])
    op.create_index("ix_retrieval_chunks_part", "retrieval_chunks", ["part"])
    op.create_index("ix_retrieval_chunks_subpart", "retrieval_chunks", ["subpart"])
    op.create_table(
        "bm25_terms",
        sa.Column("term", sa.String(length=128), primary_key=True),
        sa.Column("document_frequency", sa.Integer(), nullable=False),
        sa.Column("inverse_document_frequency", sa.Float(), nullable=False),
    )
    op.create_table(
        "bm25_postings",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("term", sa.String(length=128), sa.ForeignKey("bm25_terms.term", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_id", sa.Integer(), sa.ForeignKey("retrieval_chunks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("term_frequency", sa.Integer(), nullable=False),
        sa.UniqueConstraint("term", "chunk_id", name="uq_bm25_term_chunk"),
    )
    op.create_index("ix_bm25_postings_term", "bm25_postings", ["term"])
    op.create_index("ix_bm25_postings_chunk_id", "bm25_postings", ["chunk_id"])

    op.create_table(
        "bm25_corpus_stats",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("total_chunks", sa.Integer(), nullable=False),
        sa.Column("average_document_length", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("bm25_corpus_stats")
    op.drop_index("ix_bm25_postings_chunk_id", table_name="bm25_postings")
    op.drop_index("ix_bm25_postings_term", table_name="bm25_postings")
    op.drop_table("bm25_postings")
    op.drop_table("bm25_terms")
    op.drop_index("ix_retrieval_chunks_subpart", table_name="retrieval_chunks")
    op.drop_index("ix_retrieval_chunks_part", table_name="retrieval_chunks")
    op.drop_index("ix_retrieval_chunks_section", table_name="retrieval_chunks")
    op.drop_table("retrieval_chunks")
