"""Lexical BM25 retrieval backend."""

from __future__ import annotations

from typing import ClassVar

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import RetrievalChunk
from app.schemas.retrieval import RetrievalEvidence, StructuralFilters
from app.schemas.types import RetrievalModeEnum
from app.services.chunk_contract import build_retrieval_evidence, build_structural_filter_clauses


class BM25Service:
    """Run lexical BM25 retrieval against indexed chunk text."""

    index_name: ClassVar[str] = "retrieval_chunks_search_text_bm25_idx"

    async def search(
        self,
        session: AsyncSession,
        query_text: str,
        limit: int,
        filters: StructuralFilters | None = None,
    ) -> list[RetrievalEvidence]:
        """Return chunk evidence ranked by BM25 lexical relevance."""

        if not query_text.strip():
            return []

        bm25_query = func.to_bm25query(query_text, self.index_name)
        raw_score = RetrievalChunk.search_text.op("<@>")(bm25_query)
        fetch_limit = limit * 5 if filters else limit

        query = (
            select(RetrievalChunk, (-raw_score).label("bm25_score"))
            .where(*build_structural_filter_clauses(filters))
            .order_by(raw_score)
            .limit(fetch_limit)
        )
        rows = (await session.execute(query)).all()
        return [
            build_retrieval_evidence(
                chunk,
                retrieval_mode=RetrievalModeEnum.BM25_ONLY,
                score=float(score),
                metadata_extra={"bm25_score": float(score)},
            )
            for chunk, score in rows[:limit]
        ]
