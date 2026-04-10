"""Schemas for the main chat API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.types import QueryIntentEnum


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
    intent: QueryIntentEnum
    retrieval_rounds: int
    debug: dict[str, Any] | None = None
