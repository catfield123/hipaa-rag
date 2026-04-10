from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class StructuralFilters(BaseModel):
    part_number: str | None = None
    section_number: str | None = None
    subpart: str | None = None
    marker_path: list[str] = Field(default_factory=list)


class QueryVariant(BaseModel):
    text: str = Field(min_length=1)
    mode: Literal["bm25_only", "hybrid"]
    strategy: str
    reason: str
    filters: StructuralFilters | None = None


class AnswerConstraints(BaseModel):
    allow_negative_answer_only_with_evidence: bool = True
    require_quote_if_found: bool = True


class QueryPlan(BaseModel):
    intent: Literal[
        "general",
        "quote_request",
        "existence_check",
        "list_references",
        "ambiguous",
    ]
    queries: list[QueryVariant]
    answer_constraints: AnswerConstraints = Field(default_factory=AnswerConstraints)


class RetrievalEvidence(BaseModel):
    chunk_id: int
    source_label: str
    page_start: int
    page_end: int
    content: str
    content_with_context: str
    retrieval_mode: Literal["bm25_only", "dense", "hybrid"]
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class QuoteSpan(BaseModel):
    node_id: int
    source_label: str
    page_start: int
    page_end: int
    text: str
    char_start: int
    char_end: int


class EvidenceDecision(BaseModel):
    sufficient: bool
    rationale: str
    follow_up_actions: list[str] = Field(default_factory=list)


class SourceItem(BaseModel):
    source_label: str
    page_start: int
    page_end: int


class ChatQueryRequest(BaseModel):
    question: str = Field(min_length=3)
    include_debug: bool = False


class ChatQueryResponse(BaseModel):
    answer: str
    quotes: list[QuoteSpan] = Field(default_factory=list)
    sources: list[SourceItem] = Field(default_factory=list)
    intent: str
    retrieval_rounds: int
    debug: dict[str, Any] | None = None


class SearchRequest(BaseModel):
    query_text: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    filters: StructuralFilters | None = None


class SearchResponse(BaseModel):
    mode: Literal["bm25_only", "dense", "hybrid"]
    query_text: str
    limit: int
    filters: StructuralFilters | None = None
    results: list[RetrievalEvidence] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str


class FetchSpanRequest(BaseModel):
    node_id: int
    char_start: int = 0
    char_end: int | None = None
    expand: Literal["none", "sentence", "paragraph"] = "none"


class NodeResponse(BaseModel):
    id: int
    parent_id: int | None
    source_label: str
    heading: str | None
    raw_text: str
    page_start: int
    page_end: int


class IngestionSummary(BaseModel):
    document_nodes: int
    retrieval_chunks: int
    bm25_terms: int


class IngestionResult(BaseModel):
    status: str
    summary: IngestionSummary
