"""Schemas for system-level and operational API responses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


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
