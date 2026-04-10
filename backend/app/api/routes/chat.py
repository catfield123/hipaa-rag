from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_db_session
from app.schemas import (
    ChatQueryRequest,
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


router = APIRouter(prefix="/chat", tags=["chat"])
DbSessionDep = Annotated[AsyncSession, Depends(get_db_session)]


@router.post("/query")
async def query_chat(payload: ChatQueryRequest, session: DbSessionDep) -> ChatQueryResponse:
    settings = get_settings()
    answering_service = AnsweringService()
    retrieval_service = RetrievalService()

    plan = await answering_service.plan_queries(payload.question)
    intent = plan.intent
    evidence: list[RetrievalEvidence] = []
    decision = EvidenceDecision(
        sufficient=False,
        rationale="No retrieval evidence has been evaluated yet.",
    )
    retrieval_rounds = 0
    attempted_queries: list[QueryVariant] = []
    debug_rounds: list[dict[str, object]] = []

    for retrieval_round in range(1, settings.agent_max_rounds + 1):
        retrieval_rounds = retrieval_round
        attempted_queries.extend(plan.queries)
        round_evidence = await retrieval_service.execute_plan(session=session, plan=plan)
        evidence = _merge_evidence(evidence, round_evidence)

        decision = await answering_service.assess_evidence(
            question=payload.question,
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
        if decision.sufficient:
            break

        if retrieval_round == settings.agent_max_rounds:
            break

        next_queries = _filter_unseen_queries(decision.next_queries, attempted_queries)
        if not next_queries:
            break

        plan = QueryPlan(
            intent=intent,
            queries=next_queries,
            answer_constraints=plan.answer_constraints,
        )

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

    if decision.sufficient:
        answer = await answering_service.synthesize_answer(
            question=payload.question,
            intent=intent,
            evidence=evidence,
        )
    else:
        answer = answering_service.render_insufficient_answer(
            evidence=evidence,
            rationale=decision.rationale,
            retrieval_rounds=retrieval_rounds,
        )

    debug = None
    if payload.include_debug:
        debug = {
            "intent": intent,
            "attempted_queries": [query.model_dump() for query in attempted_queries],
            "evidence": [item.model_dump() for item in evidence],
            "final_decision": decision.model_dump(),
            "rounds": debug_rounds,
        }

    return ChatQueryResponse(
        answer=answer,
        quotes=quotes,
        sources=list(sources_map.values()),
        intent=intent,
        retrieval_rounds=retrieval_rounds,
        debug=debug,
    )


def _merge_evidence(
    existing: list[RetrievalEvidence],
    new_items: list[RetrievalEvidence],
) -> list[RetrievalEvidence]:
    merged: list[RetrievalEvidence] = []
    seen_chunk_ids: set[int] = set()
    for item in [*existing, *new_items]:
        if item.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(item.chunk_id)
        merged.append(item)
    return merged


def _filter_unseen_queries(
    candidates: list[QueryVariant],
    attempted_queries: list[QueryVariant],
) -> list[QueryVariant]:
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
