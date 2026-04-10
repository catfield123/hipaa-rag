"""Dense vector retrieval backend."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import RetrievalChunk
from app.schemas.retrieval import RetrievalEvidence, StructuralFilters
from app.services.chunk_contract import build_retrieval_evidence, build_structural_filter_clauses
from app.services.embeddings import EmbeddingService


class DenseRetriever:
    """Run dense pgvector retrieval against chunk embeddings."""

    def __init__(self, *, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service

    async def search(
        self,
        *,
        session: AsyncSession,
        query_text: str,
        limit: int,
        filters: StructuralFilters | None = None,
    ) -> list[RetrievalEvidence]:
        """Return chunk evidence ranked by vector similarity."""

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
