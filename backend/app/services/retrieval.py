from __future__ import annotations

from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import RetrievalChunk
from app.schemas import QueryPlan, RetrievalEvidence, StructuralFilters
from app.services.bm25 import BM25Service
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
        bm25_hits = await self.bm25_service.search(
            session=session,
            query_text=query_text,
            limit=self.settings.bm25_limit,
            filters=filters,
        )
        if not self.settings.openai_api_key:
            return bm25_hits[:limit]
        dense_hits = await self.dense_search(
            session=session,
            query_text=query_text,
            limit=self.settings.dense_limit,
            filters=filters,
        )
        return self._rrf_fuse([bm25_hits, dense_hits], limit=limit)

    async def dense_search(
        self,
        session: AsyncSession,
        query_text: str,
        limit: int,
        filters: StructuralFilters | None = None,
    ) -> list[RetrievalEvidence]:
        query_embedding = await self.embedding_service.embed_query(query_text)
        distance = RetrievalChunk.embedding.cosine_distance(query_embedding)
        fetch_limit = limit * 10 if filters else limit
        rows = (
            await session.execute(
                select(RetrievalChunk, distance.label("distance"))
                .where(RetrievalChunk.embedding.is_not(None))
                .order_by(distance)
                .limit(fetch_limit)
            )
        ).all()

        evidence: list[RetrievalEvidence] = []
        for chunk, distance_value in rows:
            if filters and not self._matches_filters(chunk.metadata_json, filters):
                continue
            evidence.append(
                RetrievalEvidence(
                    chunk_id=chunk.id,
                    source_label=chunk.source_label,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    content=chunk.content,
                    content_with_context=chunk.content_with_context,
                    retrieval_mode="dense",
                    score=max(0.0, 1.0 - float(distance_value)),
                    metadata=chunk.metadata_json,
                )
            )
            if len(evidence) >= limit:
                break
        return evidence

    def _rrf_fuse(
        self,
        result_sets: list[list[RetrievalEvidence]],
        limit: int,
        k: int = 60,
    ) -> list[RetrievalEvidence]:
        scores: dict[int, float] = defaultdict(float)
        evidence_by_chunk_id: dict[int, RetrievalEvidence] = {}

        for result_set in result_sets:
            for rank, evidence in enumerate(result_set, start=1):
                scores[evidence.chunk_id] += 1.0 / (k + rank)
                evidence_by_chunk_id[evidence.chunk_id] = evidence

        ranked_chunk_ids = sorted(scores, key=scores.get, reverse=True)[:limit]
        fused: list[RetrievalEvidence] = []
        for chunk_id in ranked_chunk_ids:
            base = evidence_by_chunk_id[chunk_id]
            fused.append(base.model_copy(update={"score": scores[chunk_id], "retrieval_mode": "hybrid"}))
        return fused

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

    def _matches_filters(self, metadata: dict[str, object], filters: StructuralFilters) -> bool:
        if filters.part_number:
            included_parts = {str(value) for value in metadata.get("included_part_numbers", []) if value is not None}
            fallback_part = metadata.get("part_number")
            if fallback_part is not None:
                included_parts.add(str(fallback_part))
            if filters.part_number not in included_parts:
                return False

        if filters.section_number:
            included_sections = {
                str(value) for value in metadata.get("included_section_numbers", []) if value is not None
            }
            fallback_section = metadata.get("section_number")
            if fallback_section is not None:
                included_sections.add(str(fallback_section))
            if filters.section_number not in included_sections:
                return False

        if filters.subpart:
            expected_subpart = filters.subpart.upper()
            included_subparts = {
                str(value).upper() for value in metadata.get("included_subparts", []) if value is not None
            }
            fallback_subpart = metadata.get("subpart")
            if fallback_subpart is not None:
                included_subparts.add(str(fallback_subpart).upper())
            if expected_subpart not in included_subparts:
                return False

        if filters.marker_path:
            expected_marker = f"({filters.marker_path[-1]})"
            included_markers = {str(value) for value in metadata.get("included_markers", []) if value is not None}
            fallback_marker = metadata.get("marker")
            if fallback_marker is not None:
                included_markers.add(str(fallback_marker))
            if expected_marker not in included_markers:
                return False

        return True
