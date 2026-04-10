from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import get_settings


settings = get_settings()


class Base(DeclarativeBase):
    pass


class RetrievalChunk(Base):
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
    __tablename__ = "structural_content"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    content_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
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
