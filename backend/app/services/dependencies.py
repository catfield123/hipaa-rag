"""Cached dependency providers for top-level services."""

from __future__ import annotations

from functools import lru_cache

from app.services.answering import AnsweringService
from app.services.openai_client import get_openai_client
from app.services.rag_response_builder import RagResponseBuilder


@lru_cache
def get_rag_response_builder() -> RagResponseBuilder:
    """Return the shared builder for public RAG responses."""

    return RagResponseBuilder()


@lru_cache
def get_answering_service() -> AnsweringService:
    """Return the shared answering service."""

    return AnsweringService(client=get_openai_client())
