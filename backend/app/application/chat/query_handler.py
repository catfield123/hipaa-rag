"""Chat use case orchestration for tool-driven retrieval and answering."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.schemas import ChatQueryResponse, QuoteSpan, RetrievalEvidence, SourceItem
from app.services.answering import AnsweringService
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


class ChatQueryHandler:
    """Coordinate tool-driven retrieval and final response shaping."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        answering_service: AnsweringService | None = None,
        bm25_service: BM25Service | None = None,
        dense_retriever: DenseRetriever | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        structural_retriever: StructuralContentRetriever | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.answering_service = answering_service or AnsweringService()
        self.bm25_service = bm25_service or get_bm25_service()
        self.dense_retriever = dense_retriever or get_dense_retriever()
        self.hybrid_retriever = hybrid_retriever or get_hybrid_retriever()
        self.structural_retriever = structural_retriever or get_structural_content_retriever()

    async def handle_query(
        self,
        *,
        question: str,
        include_debug: bool,
        session: AsyncSession,
    ) -> ChatQueryResponse:
        """Run the tool-driven chat workflow and return the API response payload."""

        outcome = await self.answering_service.answer_question(
            question=question,
            session=session,
            bm25_service=self.bm25_service,
            dense_retriever=self.dense_retriever,
            hybrid_retriever=self.hybrid_retriever,
            structural_retriever=self.structural_retriever,
        )
        debug_payload = None
        if include_debug:
            debug_payload = {
                "intent": outcome.intent,
                "evidence": [item.model_dump() for item in outcome.evidence],
                "rounds": outcome.debug_rounds,
            }

        return ChatQueryResponse(
            answer=outcome.answer,
            quotes=_build_quotes(outcome.evidence),
            sources=_build_sources(outcome.evidence),
            intent=outcome.intent,
            retrieval_rounds=outcome.retrieval_rounds,
            debug=debug_payload,
        )


def _build_quotes(evidence: list[RetrievalEvidence]) -> list[QuoteSpan]:
    """Build a compact quote list from the highest-ranked evidence items."""

    quotes: list[QuoteSpan] = []
    seen_chunk_ids: set[int] = set()
    for item in evidence[:3]:
        if item.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(item.chunk_id)
        quotes.append(
            QuoteSpan(
                chunk_id=item.chunk_id,
                path=item.path,
                path_text=item.path_text,
                section=item.section,
                part=item.part,
                subpart=item.subpart,
                markers=item.markers,
                text=item.text,
            )
        )
    return quotes


def _build_sources(evidence: list[RetrievalEvidence]) -> list[SourceItem]:
    """Build unique source entries for the chat response."""

    sources_map: dict[str, SourceItem] = {}
    for item in evidence[:5]:
        sources_map.setdefault(
            item.path_text,
            SourceItem(
                chunk_id=item.chunk_id,
                path_text=item.path_text,
                section=item.section,
                part=item.part,
                subpart=item.subpart,
                markers=item.markers,
            ),
        )
    return list(sources_map.values())
