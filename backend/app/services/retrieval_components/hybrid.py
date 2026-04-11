"""Hybrid lexical plus dense retrieval backend."""

from __future__ import annotations

from app.config import Settings
from app.models import RetrievalChunk
from app.schemas.retrieval import RetrievalEvidence, StructuralFilters
from app.schemas.types import RetrievalModeEnum
from app.services.chunk_contract import (
    build_retrieval_evidence,
    build_structural_filter_clauses,
)
from app.services.embeddings import EmbeddingService
from app.services.retrieval_components.bm25 import BM25Service
from sqlalchemy import func, literal, select
from sqlalchemy.ext.asyncio import AsyncSession


class HybridRetriever:
    """Fuse BM25 and dense rankings with reciprocal rank fusion (RRF); falls back to BM25 without API key."""

    def __init__(
        self,
        *,
        settings: Settings,
        embedding_service: EmbeddingService,
        bm25_service: BM25Service,
    ) -> None:
        """Wire settings and shared services for hybrid search.

        Args:
            settings (Settings): App settings (limits, RRF ``k``, API key presence).
            embedding_service (EmbeddingService): Embeds the query for the dense leg.
            bm25_service (BM25Service): Shared BM25 service (index name, search path).

        Returns:
            None

        Raises:
            None
        """

        self.settings = settings
        self.embedding_service = embedding_service
        self.bm25_service = bm25_service

    async def search(
        self,
        *,
        session: AsyncSession,
        query_text: str,
        limit: int,
        filters: StructuralFilters | None = None,
    ) -> list[RetrievalEvidence]:
        """Return chunk evidence ranked by RRF over BM25 and dense lists (BM25-only if no OpenAI key).

        Args:
            session (AsyncSession): Async SQLAlchemy session.
            query_text (str): Query text for both BM25 and embedding legs.
            limit (int): Maximum fused rows to return.
            filters (StructuralFilters | None): Structural filters applied to both legs.

        Returns:
            list[RetrievalEvidence]: Fused hits with ``retrieval_mode`` ``hybrid`` and per-leg scores in metadata.

        Raises:
            ConfigurationError: If embeddings are required but misconfigured (from the embedding service).
            sqlalchemy.exc.SQLAlchemyError: On database errors.
        """

        if not self.settings.openai_api_key:
            return await self.bm25_service.search(
                session=session,
                query_text=query_text,
                limit=limit,
                filters=filters,
            )

        query_embedding = await self.embedding_service.embed_query(query_text)
        filter_clauses = build_structural_filter_clauses(filters)
        bm25_query = func.to_bm25query(query_text, self.bm25_service.index_name)
        bm25_order_expr = RetrievalChunk.search_text.op("<@>")(bm25_query)
        dense_distance = RetrievalChunk.embedding.cosine_distance(query_embedding)

        bm25_hits = (
            select(
                RetrievalChunk.id.label("chunk_id"),
                func.row_number().over(order_by=bm25_order_expr).label("bm25_rank"),
                (-bm25_order_expr).label("bm25_score"),
            )
            .where(*filter_clauses)
            .order_by(bm25_order_expr)
            .limit(self.settings.bm25_limit)
            .cte("bm25_hits")
        )
        dense_hits = (
            select(
                RetrievalChunk.id.label("chunk_id"),
                func.row_number().over(order_by=dense_distance).label("dense_rank"),
                (literal(1.0) - dense_distance).label("dense_score"),
            )
            .where(RetrievalChunk.embedding.is_not(None), *filter_clauses)
            .order_by(dense_distance)
            .limit(self.settings.dense_limit)
            .cte("dense_hits")
        )

        rrf_k = float(self.settings.hybrid_rrf_k)
        fused = (
            select(
                func.coalesce(bm25_hits.c.chunk_id, dense_hits.c.chunk_id).label("chunk_id"),
                (
                    func.coalesce(literal(1.0) / (literal(rrf_k) + bm25_hits.c.bm25_rank), literal(0.0))
                    + func.coalesce(literal(1.0) / (literal(rrf_k) + dense_hits.c.dense_rank), literal(0.0))
                ).label("hybrid_score"),
            )
            .select_from(
                bm25_hits.join(
                    dense_hits,
                    bm25_hits.c.chunk_id == dense_hits.c.chunk_id,
                    full=True,
                )
            )
            .cte("fused_hits")
        )

        rows = (
            await session.execute(
                select(
                    RetrievalChunk,
                    fused.c.hybrid_score,
                    (-bm25_order_expr).label("bm25_score"),
                    (literal(1.0) - dense_distance).label("dense_score"),
                )
                .join(fused, fused.c.chunk_id == RetrievalChunk.id)
                .order_by(fused.c.hybrid_score.desc())
                .limit(limit)
            )
        ).all()

        return [
            build_retrieval_evidence(
                chunk,
                retrieval_mode=RetrievalModeEnum.HYBRID,
                score=float(hybrid_score),
                metadata_extra={
                    "hybrid_score": float(hybrid_score),
                    "bm25_score": float(bm25_score) if bm25_score is not None else None,
                    "dense_score": float(dense_score) if dense_score is not None else None,
                },
            )
            for chunk, hybrid_score, bm25_score, dense_score in rows
        ]
