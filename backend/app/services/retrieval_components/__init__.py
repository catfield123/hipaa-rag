"""Composable retrieval backends and dependency providers."""

from app.services.retrieval_components.bm25 import BM25Service
from app.services.retrieval_components.dense import DenseRetriever
from app.services.retrieval_components.dependencies import (
    get_bm25_service,
    get_dense_retriever,
    get_embedding_service,
    get_hybrid_retriever,
    get_structural_content_retriever,
)
from app.services.retrieval_components.hybrid import HybridRetriever
from app.services.retrieval_components.structural import StructuralContentRetriever

__all__ = [
    "BM25Service",
    "DenseRetriever",
    "HybridRetriever",
    "StructuralContentRetriever",
    "get_bm25_service",
    "get_dense_retriever",
    "get_embedding_service",
    "get_hybrid_retriever",
    "get_structural_content_retriever",
]
