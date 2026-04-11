"""Schemas for retrieval inputs, filters, and evidence payloads.

Field-level API documentation lives on ``Field(description=...)`` for ``StructuralFilters``, ``RetrievalEvidence``,
``SearchRequest``, and ``SearchResponse``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.types import RetrievalModeEnum, StructuralContentTargetEnum


class StructuralFilters(BaseModel):
    """Optional structure-aware filters used to narrow retrieval."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "part_number": "164",
                "section_number": "164.312",
                "subpart": "C",
                "marker_path": ["164.312", "a", "2", "iv"],
            }
        }
    )

    part_number: str | None = Field(default=None, description="Limit to a CFR part number string.")
    section_number: str | None = Field(default=None, description="Limit to a section id.")
    subpart: str | None = Field(default=None, description="Limit to a subpart letter or code.")
    marker_path: list[str] = Field(
        default_factory=list,
        description="Ordered marker segments for deep links.",
    )


class RetrievalEvidence(BaseModel):
    """Normalized evidence item returned by any retrieval strategy."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk_id": 1,
                "path": [],
                "path_text": "45 CFR Part 164",
                "text": "Sample chunk text…",
                "section": None,
                "part": None,
                "subpart": None,
                "markers": [],
                "retrieval_mode": "hybrid",
                "score": 0.42,
                "metadata": {},
            }
        }
    )

    chunk_id: int = Field(description="Source chunk primary key.")
    path: list[str] = Field(default_factory=list, description="Hierarchy for display.")
    path_text: str = Field(description="Flattened path label.")
    text: str = Field(description="Chunk body text.")
    section: str | None = Field(default=None, description="Section label when present.")
    part: str | None = Field(default=None, description="Part label when present.")
    subpart: str | None = Field(default=None, description="Subpart label when present.")
    markers: list[str] = Field(default_factory=list, description="Regulatory markers.")
    retrieval_mode: RetrievalModeEnum = Field(description="Backend that produced this row.")
    score: float = Field(description="Similarity or fusion score (backend-specific).")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional diagnostic fields from the retriever.",
    )


class SearchRequest(BaseModel):
    """Payload for admin retrieval debugging endpoints."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query_text": "encryption administrative safeguards",
                    "limit": 10,
                    "filters": None,
                    "structure_target": None,
                },
                {
                    "query_text": "section text",
                    "limit": 5,
                    "filters": {"part_number": "164", "section_number": None, "subpart": "C", "marker_path": []},
                    "structure_target": "section_text",
                },
            ]
        }
    )

    query_text: str = Field(
        min_length=1,
        description="Search query or placeholder text (structure lookup may ignore plain retrieval query).",
        examples=["minimum necessary standard"],
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum hits to return.",
    )
    filters: StructuralFilters | None = Field(
        default=None,
        description="Optional structural filters; applies to search modes that support them.",
    )
    structure_target: StructuralContentTargetEnum | None = Field(
        default=None,
        description="Required for `/admin/search/structure`: which structural artifact to fetch.",
    )


class SearchResponse(BaseModel):
    """Response shape for admin retrieval debugging endpoints."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "mode": "hybrid",
                "query_text": "encryption",
                "limit": 10,
                "filters": None,
                "results": [],
            }
        }
    )

    mode: RetrievalModeEnum = Field(description="Which backend produced `results`.")
    query_text: str = Field(description="Echo of the request query text.")
    limit: int = Field(description="Echo of the requested limit.")
    filters: StructuralFilters | None = Field(
        default=None,
        description="Echo of structural filters when provided.",
    )
    results: list[RetrievalEvidence] = Field(
        default_factory=list,
        description="Ranked evidence rows.",
    )
