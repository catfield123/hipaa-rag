"""Schemas for the main chat API.

Each model lists field semantics via ``Field(description=...)`` for OpenAPI; class docstrings summarize the
aggregate shape (``QuoteSpan``, ``SourceItem``, ``ChatQueryRequest``, ``ChatQueryResponse``).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.types import QueryIntentEnum


class QuoteSpan(BaseModel):
    """Quoted evidence excerpt returned to the client."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk_id": 42,
                "path": ["45 CFR Part 164", "Subpart C"],
                "path_text": "45 CFR Part 164 › Subpart C",
                "section": "§ 164.312",
                "part": "164",
                "subpart": "C",
                "markers": ["164.312(a)(2)(iv)"],
                "text": "Implement a mechanism to encrypt and decrypt electronic protected health information.",
            }
        }
    )

    chunk_id: int = Field(description="Database id of the source chunk.")
    path: list[str] = Field(
        default_factory=list,
        description="Hierarchical path segments for display.",
    )
    path_text: str = Field(description="Single-line path label for UI rendering.")
    section: str | None = Field(default=None, description="Section label when available.")
    part: str | None = Field(default=None, description="CFR part when available.")
    subpart: str | None = Field(default=None, description="Subpart when available.")
    markers: list[str] = Field(
        default_factory=list,
        description="Regulatory markers or paragraph ids attached to the chunk.",
    )
    text: str = Field(description="Verbatim excerpt text.")


class SourceItem(BaseModel):
    """Compact source metadata rendered in chat responses."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk_id": 42,
                "path_text": "45 CFR Part 164 › Subpart C",
                "section": "§ 164.312",
                "part": "164",
                "subpart": "C",
                "markers": ["164.312(a)(2)(iv)"],
            }
        }
    )

    chunk_id: int = Field(description="Database id of the source chunk.")
    path_text: str = Field(description="Single-line path label.")
    section: str | None = Field(default=None, description="Section label when available.")
    part: str | None = Field(default=None, description="CFR part when available.")
    subpart: str | None = Field(default=None, description="Subpart when available.")
    markers: list[str] = Field(
        default_factory=list,
        description="Regulatory markers attached to the chunk.",
    )


class ChatQueryRequest(BaseModel):
    """Incoming payload for the main chat endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "question": "Does HIPAA require encryption for electronic PHI at rest?",
                }
            ]
        }
    )

    question: str = Field(
        min_length=3,
        description="Natural-language question about HIPAA (minimum 3 characters).",
        examples=["Does HIPAA require encryption for electronic PHI at rest?"],
    )


class ChatQueryResponse(BaseModel):
    """Structured response returned by the chat endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "answer": "Yes. Security Rule standards address encryption of ePHI...",
                    "quotes": [],
                    "sources": [],
                    "intent": "general",
                    "retrieval_rounds": 2,
                }
            ]
        }
    )

    answer: str = Field(description="Final natural-language answer grounded in retrieved chunks.")
    quotes: list[QuoteSpan] = Field(
        default_factory=list,
        description="Quoted excerpts (one per distinct chunk, in evidence order).",
    )
    sources: list[SourceItem] = Field(
        default_factory=list,
        description="Per-chunk source lines aligned with quotes.",
    )
    intent: QueryIntentEnum = Field(description="Classifier intent for the answer path.")
    retrieval_rounds: int = Field(description="Number of retrieval rounds executed before the final answer.")
