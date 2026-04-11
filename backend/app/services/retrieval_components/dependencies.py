"""Cached dependency providers for retrieval components."""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.services.embeddings import EmbeddingService
from app.services.retrieval_components.bm25 import BM25Service
from app.services.retrieval_components.dense import DenseRetriever
from app.services.retrieval_components.hybrid import HybridRetriever
from app.services.retrieval_components.structural import StructuralContentRetriever


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Return the shared embedding service.

    Args:
        None

    Returns:
        EmbeddingService: Process-wide singleton.

    Raises:
        None
    """

    return EmbeddingService()


@lru_cache
def get_bm25_service() -> BM25Service:
    """Return the shared BM25 retrieval service.

    Args:
        None

    Returns:
        BM25Service: Process-wide singleton.

    Raises:
        None
    """

    return BM25Service()


@lru_cache
def get_structural_content_retriever() -> StructuralContentRetriever:
    """Return the shared structural content retriever.

    Args:
        None

    Returns:
        StructuralContentRetriever: Process-wide singleton.

    Raises:
        None
    """

    return StructuralContentRetriever()


@lru_cache
def get_dense_retriever() -> DenseRetriever:
    """Return the shared dense retriever.

    Args:
        None

    Returns:
        DenseRetriever: Wired with :func:`get_embedding_service`.

    Raises:
        None
    """

    return DenseRetriever(embedding_service=get_embedding_service())


@lru_cache
def get_hybrid_retriever() -> HybridRetriever:
    """Return the shared hybrid retriever.

    Args:
        None

    Returns:
        HybridRetriever: Wired with settings, embeddings, and BM25.

    Raises:
        None
    """

    return HybridRetriever(
        settings=get_settings(),
        embedding_service=get_embedding_service(),
        bm25_service=get_bm25_service(),
    )
