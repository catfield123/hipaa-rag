"""Cached dependency providers for retrieval components."""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.services.embeddings import EmbeddingService
from app.services.retrieval_components.bm25 import BM25Service
from app.services.retrieval_components.dense import DenseRetriever
from app.services.retrieval_components.hybrid import HybridRetriever
from app.services.retrieval_components.orchestrator import RetrievalOrchestrator
from app.services.retrieval_components.structural import StructuralContentRetriever


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Return the shared embedding service."""

    return EmbeddingService()


@lru_cache
def get_bm25_service() -> BM25Service:
    """Return the shared BM25 retrieval service."""

    return BM25Service()


@lru_cache
def get_structural_content_retriever() -> StructuralContentRetriever:
    """Return the shared structural content retriever."""

    return StructuralContentRetriever()


@lru_cache
def get_dense_retriever() -> DenseRetriever:
    """Return the shared dense retriever."""

    return DenseRetriever(embedding_service=get_embedding_service())


@lru_cache
def get_hybrid_retriever() -> HybridRetriever:
    """Return the shared hybrid retriever."""

    return HybridRetriever(
        settings=get_settings(),
        embedding_service=get_embedding_service(),
        bm25_service=get_bm25_service(),
    )


@lru_cache
def get_retrieval_orchestrator() -> RetrievalOrchestrator:
    """Return the shared orchestrator for retrieval plans."""

    return RetrievalOrchestrator(
        settings=get_settings(),
        bm25_service=get_bm25_service(),
        hybrid_retriever=get_hybrid_retriever(),
        structural_retriever=get_structural_content_retriever(),
    )
