"""SQLAlchemy ORM models mirroring :mod:`alembic.versions.0001_initial` (chunks and structural content)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.config import get_settings
from app.schemas.types import StructuralContentTargetEnum
from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

settings = get_settings()


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for HIPAA RAG ORM models."""



class RetrievalChunk(Base):
    """Single searchable chunk row (lexical ``search_text``, optional embedding vector).

    Args (columns):
        id (int): Chunk primary key (matches JSON chunk ``id`` from ingestion).
        path (list[str]): Hierarchical path segments (JSONB).
        path_text (str): Display path string.
        text (str): Chunk body.
        search_text (str): Generated ``path_text || newline || text`` for BM25.
        section, part, subpart (str | None): Regulatory labels for filtering.
        markers (list[str]): Parenthetical markers (JSONB array).
        token_count (int): Approximate token count for metrics.
        metadata_json (dict): Provenance and extras (JSONB).
        embedding (list[float] | None): pgvector embedding or ``None``.
        created_at (datetime): Insert timestamp (server default).
    """

    __tablename__ = "retrieval_chunks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    path: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    path_text: Mapped[str] = mapped_column(Text, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    search_text: Mapped[str] = mapped_column(
        Text,
        Computed("coalesce(path_text, '') || E'\\n' || coalesce(text, '')", persisted=True),
        nullable=False,
    )
    section: Mapped[str | None] = mapped_column(String(255), index=True)
    part: Mapped[str | None] = mapped_column(String(255), index=True)
    subpart: Mapped[str | None] = mapped_column(String(255), index=True)
    markers: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.embedding_dimension),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class StructuralContent(Base):
    """Precomputed section text or outline material for direct structural retrieval.

    Args (columns):
        id (int): Surrogate key.
        content_type (StructuralContentTargetEnum): Section vs outline variant.
        path, path_text, text: Location and body text.
        part, subpart, section: Human-readable labels.
        part_number, subpart_key, section_number: Normalized filter keys.
        metadata_json (dict): Outline structure or section title metadata.
        created_at (datetime): Insert timestamp.
    """

    __tablename__ = "structural_content"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    content_type: Mapped[StructuralContentTargetEnum] = mapped_column(String(32), nullable=False, index=True)
    path: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    path_text: Mapped[str] = mapped_column(Text, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    part: Mapped[str | None] = mapped_column(String(255), index=True)
    subpart: Mapped[str | None] = mapped_column(String(255), index=True)
    section: Mapped[str | None] = mapped_column(String(255), index=True)
    part_number: Mapped[str | None] = mapped_column(String(32), index=True)
    subpart_key: Mapped[str | None] = mapped_column(String(32), index=True)
    section_number: Mapped[str | None] = mapped_column(String(32), index=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
