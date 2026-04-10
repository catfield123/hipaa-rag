"""Decision models for multi-round retrieval orchestration."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas import QueryIntent


class ResearchDecision(BaseModel):
    """Decision returned after inspecting the evidence gathered so far."""

    intent: QueryIntent
    wants_raw_structure: bool
    continue_retrieval: bool
    rationale: str
    missing_information: list[str] = Field(default_factory=list)
