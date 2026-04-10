from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import get_settings


settings = get_settings()


class Base(DeclarativeBase):
    pass


class NodeType(str, Enum):
    part = "part"
    subpart = "subpart"
    section = "section"
    paragraph = "paragraph"
    subparagraph = "subparagraph"
    text = "text"


class RetrievalMode(str, Enum):
    bm25_only = "bm25_only"
    hybrid = "hybrid"


class DocumentNode(Base):
    __tablename__ = "document_nodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("document_nodes.id", ondelete="CASCADE"),
        nullable=True,
    )
    node_type: Mapped[str] = mapped_column(String(32), nullable=False)
    part_number: Mapped[str | None] = mapped_column(String(16))
    subpart: Mapped[str | None] = mapped_column(String(32))
    section_number: Mapped[str | None] = mapped_column(String(32))
    marker: Mapped[str | None] = mapped_column(String(32))
    source_label: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    heading: Mapped[str | None] = mapped_column(String(512))
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    parent: Mapped[DocumentNode | None] = relationship(
        back_populates="children",
        remote_side="DocumentNode.id",
    )
    children: Mapped[list[DocumentNode]] = relationship(
        back_populates="parent",
        cascade="all, delete-orphan",
    )


class RetrievalChunk(Base):
    __tablename__ = "retrieval_chunks"
    __table_args__ = (
        UniqueConstraint("start_node_id", "chunk_index", name="uq_chunk_start_node_index"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    start_node_id: Mapped[int] = mapped_column(
        ForeignKey("document_nodes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    end_node_id: Mapped[int] = mapped_column(
        ForeignKey("document_nodes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    source_label: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_with_context: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.embedding_dimension),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    start_node: Mapped[DocumentNode] = relationship(foreign_keys=[start_node_id])
    end_node: Mapped[DocumentNode] = relationship(foreign_keys=[end_node_id])


class BM25Term(Base):
    __tablename__ = "bm25_terms"

    term: Mapped[str] = mapped_column(String(128), primary_key=True)
    document_frequency: Mapped[int] = mapped_column(Integer, nullable=False)
    inverse_document_frequency: Mapped[float] = mapped_column(Float, nullable=False)


class BM25Posting(Base):
    __tablename__ = "bm25_postings"
    __table_args__ = (
        UniqueConstraint("term", "chunk_id", name="uq_bm25_term_chunk"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    term: Mapped[str] = mapped_column(
        ForeignKey("bm25_terms.term", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("retrieval_chunks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    term_frequency: Mapped[int] = mapped_column(Integer, nullable=False)


class BM25CorpusStat(Base):
    __tablename__ = "bm25_corpus_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    total_chunks: Mapped[int] = mapped_column(Integer, nullable=False)
    average_document_length: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
