from __future__ import annotations

from sqlalchemy import func, literal, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import RetrievalChunk
from app.schemas import QueryPlan, RetrievalEvidence, StructuralFilters
from app.services.bm25 import BM25Service
from app.services.chunk_contract import build_retrieval_evidence, build_structural_filter_clauses
from app.services.embeddings import EmbeddingService
from app.services.text_utils import unique_preserve_order


class RetrievalService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedding_service = EmbeddingService()
        self.bm25_service = BM25Service()

    async def execute_plan(
        self,
        session: AsyncSession,
        plan: QueryPlan,
    ) -> list[RetrievalEvidence]:
        evidence_sets: list[list[RetrievalEvidence]] = []
        for variant in plan.queries:
            if variant.mode == "bm25_only":
                evidence = await self.bm25_service.search(
                    session=session,
                    query_text=variant.text,
                    limit=self.settings.bm25_limit,
                    filters=variant.filters,
                )
            else:
                evidence = await self.hybrid_search(
                    session=session,
                    query_text=variant.text,
                    limit=self.settings.retrieval_limit,
                    filters=variant.filters,
                )
            evidence_sets.append(evidence)

        return self._merge_evidence_sets(evidence_sets, limit=self.settings.retrieval_limit)

    async def hybrid_search(
        self,
        session: AsyncSession,
        query_text: str,
        limit: int,
        filters: StructuralFilters | None = None,
    ) -> list[RetrievalEvidence]:
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
                bm25_hits.c.bm25_score.label("bm25_score"),
                dense_hits.c.dense_score.label("dense_score"),
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
                    fused.c.bm25_score,
                    fused.c.dense_score,
                )
                .join(fused, fused.c.chunk_id == RetrievalChunk.id)
                .order_by(fused.c.hybrid_score.desc())
                .limit(limit)
            )
        ).all()

        return [
            build_retrieval_evidence(
                chunk,
                retrieval_mode="hybrid",
                score=float(hybrid_score),
                metadata_extra={
                    "hybrid_score": float(hybrid_score),
                    "bm25_score": float(bm25_score) if bm25_score is not None else None,
                    "dense_score": float(dense_score) if dense_score is not None else None,
                },
            )
            for chunk, hybrid_score, bm25_score, dense_score in rows
        ]

    async def dense_search(
        self,
        session: AsyncSession,
        query_text: str,
        limit: int,
        filters: StructuralFilters | None = None,
    ) -> list[RetrievalEvidence]:
        query_embedding = await self.embedding_service.embed_query(query_text)
        distance = RetrievalChunk.embedding.cosine_distance(query_embedding)
        fetch_limit = limit * 5 if filters else limit
        rows = (
            await session.execute(
                select(RetrievalChunk, distance.label("distance"))
                .where(RetrievalChunk.embedding.is_not(None), *build_structural_filter_clauses(filters))
                .order_by(distance)
                .limit(fetch_limit)
            )
        ).all()

        return [
            build_retrieval_evidence(
                chunk,
                retrieval_mode="dense",
                score=max(0.0, 1.0 - float(distance_value)),
                metadata_extra={"dense_score": max(0.0, 1.0 - float(distance_value))},
            )
            for chunk, distance_value in rows[:limit]
        ]

    def _merge_evidence_sets(
        self,
        evidence_sets: list[list[RetrievalEvidence]],
        limit: int,
    ) -> list[RetrievalEvidence]:
        merged_order = unique_preserve_order(
            str(evidence.chunk_id)
            for evidence_set in evidence_sets
            for evidence in evidence_set
        )
        evidence_by_id = {
            evidence.chunk_id: evidence
            for evidence_set in evidence_sets
            for evidence in evidence_set
        }
        return [
            evidence_by_id[int(chunk_id)]
            for chunk_id in merged_order[:limit]
        ]
