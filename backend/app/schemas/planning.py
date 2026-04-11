"""Schemas for research decisions used by the answering agent."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.types import QueryIntentEnum


class ResearchDecision(BaseModel):
    """Decision returned after inspecting the evidence gathered so far.

    Args (fields):
        intent (QueryIntentEnum): High-level intent for the answer path.
        wants_raw_structure (bool): Whether the user asked for structural outlines or raw hierarchy.
        continue_retrieval (bool): Whether another retrieval round is needed.
        rationale (str): Short explanation for logging and UI status.
        missing_information (list[str]): Bullet list of gaps if retrieval should continue.
    """

    intent: QueryIntentEnum = Field(description="High-level intent for the answer path.")
    wants_raw_structure: bool = Field(
        description="Whether the user asked for structural outlines or raw hierarchy.",
    )
    continue_retrieval: bool = Field(description="Whether another retrieval round is needed.")
    rationale: str = Field(description="Short explanation for logging and UI status.")
    missing_information: list[str] = Field(
        default_factory=list,
        description="Bullet list of gaps if retrieval should continue.",
    )
