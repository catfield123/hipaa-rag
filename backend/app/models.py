from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

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
    dense = "dense"
    hybrid = "hybrid"


class RetrievalChunk(Base):
    __tablename__ = "retrieval_chunks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    path: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    path_text: Mapped[str] = mapped_column(Text, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    section: Mapped[str | None] = mapped_column(String(255), index=True)
    part: Mapped[str | None] = mapped_column(String(255), index=True)
    subpart: Mapped[str | None] = mapped_column(String(255), index=True)
    markers: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
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
