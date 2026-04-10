"""Facade service for structural, BM25, dense, and hybrid retrieval."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.schemas import QueryPlan, RetrievalEvidence, StructuralContentTarget, StructuralFilters
from app.services.embeddings import EmbeddingService
from app.services.retrieval_components import (
    BM25Service,
    DenseRetriever,
    HybridRetriever,
    RetrievalOrchestrator,
    StructuralContentRetriever,
)


class RetrievalService:
    """Thin facade over decomposed retrieval backends and orchestration."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedding_service = EmbeddingService()
        self.bm25_service = BM25Service()
        self.structural_retriever = StructuralContentRetriever()
        self.dense_retriever = DenseRetriever(embedding_service=self.embedding_service)
        self.hybrid_retriever = HybridRetriever(
            settings=self.settings,
            embedding_service=self.embedding_service,
            bm25_service=self.bm25_service,
        )
        self.orchestrator = RetrievalOrchestrator(
            settings=self.settings,
            bm25_service=self.bm25_service,
            hybrid_retriever=self.hybrid_retriever,
            structural_retriever=self.structural_retriever,
        )

    async def execute_plan(
        self,
        session: AsyncSession,
        plan: QueryPlan,
    ) -> list[RetrievalEvidence]:
        """Execute a retrieval plan across the available retrieval backends."""

        return await self.orchestrator.execute_plan(session=session, plan=plan)

    async def lookup_structural_content(
        self,
        session: AsyncSession,
        target: StructuralContentTarget,
        limit: int,
        filters: StructuralFilters | None = None,
    ) -> list[RetrievalEvidence]:
        """Fetch precomputed structural content such as outlines or full section text."""

        return await self.structural_retriever.lookup(
            session=session,
            target=target,
            limit=limit,
            filters=filters,
        )

    async def hybrid_search(
        self,
        session: AsyncSession,
        query_text: str,
        limit: int,
        filters: StructuralFilters | None = None,
    ) -> list[RetrievalEvidence]:
        """Run hybrid lexical plus dense retrieval."""

        return await self.hybrid_retriever.search(
            session=session,
            query_text=query_text,
            limit=limit,
            filters=filters,
        )

    async def dense_search(
        self,
        session: AsyncSession,
        query_text: str,
        limit: int,
        filters: StructuralFilters | None = None,
    ) -> list[RetrievalEvidence]:
        """Run dense vector retrieval."""

        return await self.dense_retriever.search(
            session=session,
            query_text=query_text,
            limit=limit,
            filters=filters,
        )
