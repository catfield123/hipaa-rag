"""Chat use case orchestration for multi-round retrieval and answering."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.schemas import (
    ChatQueryResponse,
    EvidenceDecision,
    QueryPlan,
    QueryVariant,
    QuoteSpan,
    RetrievalEvidence,
    SourceItem,
)
from app.services.answering import AnsweringService
from app.services.retrieval import RetrievalService


class ChatQueryHandler:
    """Coordinate planning, retrieval, evidence judging, and answer synthesis."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        answering_service: AnsweringService | None = None,
        retrieval_service: RetrievalService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.answering_service = answering_service or AnsweringService()
        self.retrieval_service = retrieval_service or RetrievalService()

    async def handle_query(
        self,
        *,
        question: str,
        include_debug: bool,
        session: AsyncSession,
    ) -> ChatQueryResponse:
        """Run the full chat query workflow and return the API response payload."""

        plan = await self.answering_service.plan_queries(question)
        intent = plan.intent
        evidence: list[RetrievalEvidence] = []
        decision = EvidenceDecision(
            sufficient=False,
            rationale="No retrieval evidence has been evaluated yet.",
        )
        retrieval_rounds = 0
        attempted_queries: list[QueryVariant] = []
        debug_rounds: list[dict[str, object]] = []

        for retrieval_round in range(1, self.settings.agent_max_rounds + 1):
            retrieval_rounds = retrieval_round
            attempted_queries.extend(plan.queries)
            round_evidence = await self.retrieval_service.execute_plan(
                session=session,
                plan=plan,
            )
            evidence = _merge_evidence(existing=evidence, new_items=round_evidence)
            decision = await self.answering_service.assess_evidence(
                question=question,
                intent=intent,
                evidence=evidence,
                attempted_queries=attempted_queries,
                retrieval_round=retrieval_round,
            )

            debug_rounds.append(
                {
                    "round": retrieval_round,
                    "queries": [query.model_dump() for query in plan.queries],
                    "new_evidence": [item.model_dump() for item in round_evidence],
                    "total_evidence_count": len(evidence),
                    "decision": decision.model_dump(),
                }
            )
            if decision.sufficient or retrieval_round == self.settings.agent_max_rounds:
                break

            next_queries = _filter_unseen_queries(
                candidates=decision.next_queries,
                attempted_queries=attempted_queries,
            )
            if not next_queries:
                break

            plan = QueryPlan(
                intent=intent,
                queries=next_queries,
                answer_constraints=plan.answer_constraints,
            )

        answer = await self._build_answer(
            question=question,
            intent=intent,
            evidence=evidence,
            decision=decision,
            retrieval_rounds=retrieval_rounds,
        )
        debug_payload = None
        if include_debug:
            debug_payload = {
                "intent": intent,
                "attempted_queries": [query.model_dump() for query in attempted_queries],
                "evidence": [item.model_dump() for item in evidence],
                "final_decision": decision.model_dump(),
                "rounds": debug_rounds,
            }

        return ChatQueryResponse(
            answer=answer,
            quotes=_build_quotes(evidence),
            sources=_build_sources(evidence),
            intent=intent,
            retrieval_rounds=retrieval_rounds,
            debug=debug_payload,
        )

    async def _build_answer(
        self,
        *,
        question: str,
        intent: str,
        evidence: list[RetrievalEvidence],
        decision: EvidenceDecision,
        retrieval_rounds: int,
    ) -> str:
        """Return either a synthesized answer or an insufficient-evidence message."""

        if decision.sufficient:
            return await self.answering_service.synthesize_answer(
                question=question,
                intent=intent,
                evidence=evidence,
            )
        return self.answering_service.render_insufficient_answer(
            evidence=evidence,
            rationale=decision.rationale,
            retrieval_rounds=retrieval_rounds,
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


def _merge_evidence(
    *,
    existing: list[RetrievalEvidence],
    new_items: list[RetrievalEvidence],
) -> list[RetrievalEvidence]:
    """Merge evidence while preserving first-seen order by chunk id."""

    merged: list[RetrievalEvidence] = []
    seen_chunk_ids: set[int] = set()
    for item in [*existing, *new_items]:
        if item.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(item.chunk_id)
        merged.append(item)
    return merged


def _filter_unseen_queries(
    *,
    candidates: list[QueryVariant],
    attempted_queries: list[QueryVariant],
) -> list[QueryVariant]:
    """Drop follow-up queries that were already attempted in earlier rounds."""

    attempted_signatures = {_query_signature(query) for query in attempted_queries}
    filtered: list[QueryVariant] = []
    seen_signatures = set(attempted_signatures)
    for query in candidates:
        signature = _query_signature(query)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        filtered.append(query)
    return filtered


def _query_signature(
    query: QueryVariant,
) -> tuple[str, str, str | None, str | None, str | None, str | None, tuple[str, ...]]:
    """Build a stable signature for deduplicating planned queries."""

    filters = query.filters
    return (
        " ".join(query.text.lower().split()),
        query.mode,
        query.structure_target,
        filters.part_number if filters else None,
        filters.section_number if filters else None,
        filters.subpart if filters else None,
        tuple(filters.marker_path) if filters else (),
    )
