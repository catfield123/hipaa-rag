from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_db_session
from app.schemas import ChatQueryRequest, ChatQueryResponse, QuoteSpan, SourceItem
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
    evidence = []
    retrieval_rounds = 0
    failed_queries: list[str] = []

    for retrieval_round in range(1, settings.agent_max_rounds + 1):
        retrieval_rounds = retrieval_round
        evidence = await retrieval_service.execute_plan(session=session, plan=plan)

        decision = await answering_service.judge_evidence(
            question=payload.question,
            intent=plan.intent,
            evidence=evidence,
        )
        if decision.sufficient or retrieval_round == settings.agent_max_rounds:
            break

        failed_queries.extend(query.text for query in plan.queries)
        plan = await answering_service.plan_queries(
            payload.question,
            retrieval_round=retrieval_round + 1,
            previous_failed_queries=failed_queries,
            intent_hint=plan.intent,
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

    answer = await answering_service.synthesize_answer(
        question=payload.question,
        intent=plan.intent,
        evidence=evidence,
    )
    debug = None
    if payload.include_debug:
        debug = {
            "intent": plan.intent,
            "queries": [query.model_dump() for query in plan.queries],
            "evidence": [item.model_dump() for item in evidence],
        }

    return ChatQueryResponse(
        answer=answer,
        quotes=quotes,
        sources=list(sources_map.values()),
        intent=plan.intent,
        retrieval_rounds=retrieval_rounds,
        debug=debug,
    )
