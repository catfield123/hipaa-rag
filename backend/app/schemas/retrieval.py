"""Schemas for retrieval inputs, filters, and evidence payloads."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.types import RetrievalMode, StructuralContentTarget


class StructuralFilters(BaseModel):
    """Optional structure-aware filters used to narrow retrieval."""

    part_number: str | None = None
    section_number: str | None = None
    subpart: str | None = None
    marker_path: list[str] = Field(default_factory=list)


class RetrievalEvidence(BaseModel):
    """Normalized evidence item returned by any retrieval strategy."""

    chunk_id: int
    path: list[str] = Field(default_factory=list)
    path_text: str
    text: str
    section: str | None = None
    part: str | None = None
    subpart: str | None = None
    markers: list[str] = Field(default_factory=list)
    retrieval_mode: RetrievalMode
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    """Payload for admin retrieval debugging endpoints."""

    query_text: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    filters: StructuralFilters | None = None
    structure_target: StructuralContentTarget | None = None


class SearchResponse(BaseModel):
    """Response shape for admin retrieval debugging endpoints."""

    mode: RetrievalMode
    query_text: str
    limit: int
    filters: StructuralFilters | None = None
    results: list[RetrievalEvidence] = Field(default_factory=list)
