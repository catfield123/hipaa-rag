"""Schemas for system-level and operational API responses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Simple health-check response.

    Args (fields):
        status (Literal['ok']): Fixed liveness token for load balancers.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
            }
        }
    )

    status: Literal["ok"] = Field(description="Liveness indicator; always `ok` when the process responds.")


class IngestionSummary(BaseModel):
    """Summary of a successful ingestion run.

    Args (fields):
        retrieval_chunks (int): Count of persisted chunk rows.
        lexical_index (Literal['pg_textsearch']): Label for the BM25 / textsearch index.
        dense_index (Literal['pgvector_exact']): Label for the vector index mode.
        source_mode (Literal['markdown']): Ingestion source format.
    """

    retrieval_chunks: int = Field(description="Number of chunks stored for retrieval.")
    lexical_index: Literal["pg_textsearch"] = Field(description="Lexical index backend label.")
    dense_index: Literal["pgvector_exact"] = Field(description="Vector index backend label.")
    source_mode: Literal["markdown"] = Field(description="Source document format used for ingestion.")


class IngestionResult(BaseModel):
    """Response returned after ingestion completes.

    Args (fields):
        status (str): Completion label (e.g. ``completed``).
        summary (IngestionSummary): Numeric and index summary.
    """

    status: str = Field(description="Run outcome label (e.g. completed).")
    summary: IngestionSummary = Field(description="Chunk counts and index labels.")
