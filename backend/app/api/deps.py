"""Shared FastAPI dependency type aliases (Annotated + Depends).

These aliases keep route signatures short and ensure the same providers are used
for HTTP and WebSocket endpoints.

Exports:
    DbSessionDep:
        ``Annotated[AsyncSession, Depends(get_db_session)]`` — request-scoped async ORM session.
    Bm25ServiceDep:
        ``Annotated[BM25Service, Depends(get_bm25_service)]`` — shared BM25 retriever.
    DenseRetrieverDep:
        ``Annotated[DenseRetriever, Depends(get_dense_retriever)]`` — dense pgvector retriever.
    HybridRetrieverDep:
        ``Annotated[HybridRetriever, Depends(get_hybrid_retriever)]`` — hybrid fusion retriever.
    StructuralRetrieverDep:
        ``Annotated[StructuralContentRetriever, Depends(get_structural_content_retriever)]``.
    AnsweringServiceDep:
        ``Annotated[AnsweringService, Depends(get_answering_service)]`` — function-calling agent.
    RagResponseBuilderDep:
        ``Annotated[RagResponseBuilder, Depends(get_rag_response_builder)]`` — quote/source assembly.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db_session
from app.services.answering import AnsweringService
from app.services.dependencies import get_answering_service, get_rag_response_builder
from app.services.rag_response_builder import RagResponseBuilder
from app.services.retrieval_components import (
    BM25Service,
    DenseRetriever,
    HybridRetriever,
    StructuralContentRetriever,
)
from app.services.retrieval_components.dependencies import (
    get_bm25_service,
    get_dense_retriever,
    get_hybrid_retriever,
    get_structural_content_retriever,
)

DbSessionDep = Annotated[AsyncSession, Depends(get_db_session)]
Bm25ServiceDep = Annotated[BM25Service, Depends(get_bm25_service)]
DenseRetrieverDep = Annotated[DenseRetriever, Depends(get_dense_retriever)]
HybridRetrieverDep = Annotated[HybridRetriever, Depends(get_hybrid_retriever)]
StructuralRetrieverDep = Annotated[
    StructuralContentRetriever,
    Depends(get_structural_content_retriever),
]
AnsweringServiceDep = Annotated[AnsweringService, Depends(get_answering_service)]
RagResponseBuilderDep = Annotated[RagResponseBuilder, Depends(get_rag_response_builder)]
