"""Shared API and service-layer schemas for the HIPAA RAG backend."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

StructuralContentTarget = Literal["section_text", "part_outline", "subpart_outline"]
QueryMode = Literal["bm25_only", "hybrid", "structure_lookup"]
RetrievalMode = Literal["bm25_only", "dense", "hybrid", "structure_lookup"]
QueryIntent = Literal[
    "general",
    "quote_request",
    "existence_check",
    "list_references",
    "ambiguous",
    "structure_lookup",
]


class StructuralFilters(BaseModel):
    """Optional structure-aware filters used to narrow retrieval."""

    part_number: str | None = None
    section_number: str | None = None
    subpart: str | None = None
    marker_path: list[str] = Field(default_factory=list)


class QueryVariant(BaseModel):
    """A single retrieval query candidate produced by the planner."""

    text: str = Field(min_length=1)
    mode: QueryMode
    strategy: str
    reason: str
    filters: StructuralFilters | None = None
    structure_target: StructuralContentTarget | None = None


class AnswerConstraints(BaseModel):
    """Constraints that downstream answer generation should respect."""

    allow_negative_answer_only_with_evidence: bool = True
    require_quote_if_found: bool = True


class QueryPlan(BaseModel):
    """Planner output describing retrieval intent and candidate queries."""

    intent: QueryIntent
    queries: list[QueryVariant]
    answer_constraints: AnswerConstraints = Field(default_factory=AnswerConstraints)


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


class QuoteSpan(BaseModel):
    """Quoted evidence excerpt returned to the client."""

    chunk_id: int
    path: list[str] = Field(default_factory=list)
    path_text: str
    section: str | None = None
    part: str | None = None
    subpart: str | None = None
    markers: list[str] = Field(default_factory=list)
    text: str


class EvidenceDecision(BaseModel):
    """Decision describing whether the current evidence set is sufficient."""

    sufficient: bool
    rationale: str
    missing_information: list[str] = Field(default_factory=list)
    next_queries: list[QueryVariant] = Field(default_factory=list)


class SourceItem(BaseModel):
    """Compact source metadata rendered in chat responses."""

    chunk_id: int
    path_text: str
    section: str | None = None
    part: str | None = None
    subpart: str | None = None
    markers: list[str] = Field(default_factory=list)


class ChatQueryRequest(BaseModel):
    """Incoming payload for the main chat endpoint."""

    question: str = Field(min_length=3)
    include_debug: bool = False


class ChatQueryResponse(BaseModel):
    """Structured response returned by the chat endpoint."""

    answer: str
    quotes: list[QuoteSpan] = Field(default_factory=list)
    sources: list[SourceItem] = Field(default_factory=list)
    intent: QueryIntent
    retrieval_rounds: int
    debug: dict[str, Any] | None = None


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


class HealthResponse(BaseModel):
    """Simple health-check response."""

    status: Literal["ok"]


class IngestionSummary(BaseModel):
    """Summary of a successful ingestion run."""

    retrieval_chunks: int
    lexical_index: Literal["pg_textsearch"]
    dense_index: Literal["pgvector_exact"]
    source_mode: Literal["markdown"]


class IngestionResult(BaseModel):
    """Response returned after ingestion completes."""

    status: str
    summary: IngestionSummary
