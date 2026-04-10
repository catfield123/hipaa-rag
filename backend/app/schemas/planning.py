"""Schemas for retrieval planning and evidence sufficiency decisions."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.retrieval import StructuralFilters
from app.schemas.types import QueryIntent, QueryMode, StructuralContentTarget


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


class EvidenceDecision(BaseModel):
    """Decision describing whether the current evidence set is sufficient."""

    sufficient: bool
    rationale: str
    missing_information: list[str] = Field(default_factory=list)
    next_queries: list[QueryVariant] = Field(default_factory=list)


class ResearchDecision(BaseModel):
    """Decision returned after inspecting the evidence gathered so far."""

    intent: QueryIntent
    wants_raw_structure: bool
    continue_retrieval: bool
    rationale: str
    missing_information: list[str] = Field(default_factory=list)
