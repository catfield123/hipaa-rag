"""High-level orchestration for executing retrieval plans."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.schemas.planning import QueryPlan
from app.schemas.retrieval import RetrievalEvidence
from app.schemas.types import QueryModeEnum
from app.services.retrieval_components.bm25 import BM25Service
from app.services.retrieval_components.hybrid import HybridRetriever
from app.services.retrieval_components.structural import StructuralContentRetriever
from app.services.text_utils import unique_preserve_order


class RetrievalOrchestrator:
    """Dispatch each planned query to the appropriate retrieval backend."""

    def __init__(
        self,
        *,
        settings: Settings,
        bm25_service: BM25Service,
        hybrid_retriever: HybridRetriever,
        structural_retriever: StructuralContentRetriever,
    ) -> None:
        self.settings = settings
        self.bm25_service = bm25_service
        self.hybrid_retriever = hybrid_retriever
        self.structural_retriever = structural_retriever

    async def execute_plan(
        self,
        *,
        session: AsyncSession,
        plan: QueryPlan,
    ) -> list[RetrievalEvidence]:
        """Execute a query plan and merge results from all retrieval modes."""

        evidence_sets: list[list[RetrievalEvidence]] = []
        for variant in plan.queries:
            if variant.mode == QueryModeEnum.STRUCTURE_LOOKUP and variant.structure_target:
                evidence = await self.structural_retriever.lookup(
                    session=session,
                    target=variant.structure_target,
                    limit=self.settings.retrieval_limit,
                    filters=variant.filters,
                )
            elif variant.mode == QueryModeEnum.BM25_ONLY:
                evidence = await self.bm25_service.search(
                    session=session,
                    query_text=variant.text,
                    limit=self.settings.bm25_limit,
                    filters=variant.filters,
                )
            else:
                evidence = await self.hybrid_retriever.search(
                    session=session,
                    query_text=variant.text,
                    limit=self.settings.retrieval_limit,
                    filters=variant.filters,
                )
            evidence_sets.append(evidence)

        return self.merge_evidence_sets(
            evidence_sets=evidence_sets,
            limit=self.settings.retrieval_limit,
        )

    def merge_evidence_sets(
        self,
        *,
        evidence_sets: list[list[RetrievalEvidence]],
        limit: int,
    ) -> list[RetrievalEvidence]:
        """Merge evidence from multiple retrieval passes while keeping stable order."""

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
        return [evidence_by_id[int(chunk_id)] for chunk_id in merged_order[:limit]]
