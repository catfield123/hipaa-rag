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
        "document_nodes",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("parent_id", sa.Integer(), sa.ForeignKey("document_nodes.id", ondelete="CASCADE")),
        sa.Column("node_type", sa.String(length=32), nullable=False),
        sa.Column("part_number", sa.String(length=16)),
        sa.Column("subpart", sa.String(length=32)),
        sa.Column("section_number", sa.String(length=32)),
        sa.Column("marker", sa.String(length=32)),
        sa.Column("source_label", sa.String(length=255), nullable=False),
        sa.Column("heading", sa.Text()),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=False),
        sa.Column("page_end", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_document_nodes_source_label", "document_nodes", ["source_label"])

    op.create_table(
        "retrieval_chunks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("start_node_id", sa.Integer(), sa.ForeignKey("document_nodes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("end_node_id", sa.Integer(), sa.ForeignKey("document_nodes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("source_label", sa.String(length=255), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_with_context", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=False),
        sa.Column("page_end", sa.Integer(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("embedding", Vector(1536)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("start_node_id", "chunk_index", name="uq_chunk_start_node_index"),
    )
    op.create_index("ix_retrieval_chunks_start_node_id", "retrieval_chunks", ["start_node_id"])
    op.create_index("ix_retrieval_chunks_end_node_id", "retrieval_chunks", ["end_node_id"])
    op.create_index("ix_retrieval_chunks_source_label", "retrieval_chunks", ["source_label"])
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
    op.drop_index("ix_retrieval_chunks_source_label", table_name="retrieval_chunks")
    op.drop_index("ix_retrieval_chunks_end_node_id", table_name="retrieval_chunks")
    op.drop_index("ix_retrieval_chunks_start_node_id", table_name="retrieval_chunks")
    op.drop_table("retrieval_chunks")
    op.drop_index("ix_document_nodes_source_label", table_name="document_nodes")
    op.drop_table("document_nodes")
