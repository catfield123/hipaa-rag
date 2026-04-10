from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_db_session
from app.schemas import ChatQueryRequest, ChatQueryResponse, SourceItem
from app.services.answering import AnsweringService
from app.services.node_fetcher import NodeFetcher
from app.services.retrieval import RetrievalService


router = APIRouter(prefix="/chat", tags=["chat"])
DbSessionDep = Annotated[AsyncSession, Depends(get_db_session)]


@router.post("/query")
async def query_chat(payload: ChatQueryRequest, session: DbSessionDep) -> ChatQueryResponse:
    settings = get_settings()
    answering_service = AnsweringService()
    retrieval_service = RetrievalService()
    node_fetcher = NodeFetcher()

    plan = await answering_service.plan_queries(payload.question)
    evidence = []
    exact_phrase_hits: list[dict[str, int | str]] = []
    retrieval_rounds = 0
    failed_queries: list[str] = []

    for retrieval_round in range(1, settings.agent_max_rounds + 1):
        retrieval_rounds = retrieval_round
        evidence = await retrieval_service.execute_plan(session=session, plan=plan)
        if plan.needs_exact_phrase_check:
            exact_phrase_hits = await retrieval_service.exact_phrase_search(
                session=session,
                phrase=plan.queries[0].text,
            )

        decision = await answering_service.judge_evidence(
            question=payload.question,
            intent=plan.intent,
            evidence=evidence,
            exact_phrase_hits=exact_phrase_hits,
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

    quotes = []
    seen_node_ids: set[int] = set()
    for item in evidence[:3]:
        node_id = int(item.metadata.get("quote_node_id", item.metadata.get("start_node_id", 0)))
        if not node_id or node_id in seen_node_ids:
            continue
        seen_node_ids.add(node_id)
        expand = "paragraph" if plan.intent == "quote_request" else "sentence"
        quotes.append(
            await node_fetcher.get_span(
                session=session,
                node_id=node_id,
                char_start=int(item.metadata.get("char_start", item.metadata.get("chunk_char_start", 0))),
                char_end=int(item.metadata.get("char_end", item.metadata.get("chunk_char_end", 0))),
                expand=expand,
            )
        )

    if not quotes:
        for hit in exact_phrase_hits[:3]:
            node_id = int(hit["node_id"])
            if node_id in seen_node_ids:
                continue
            seen_node_ids.add(node_id)
            quotes.append(
                await node_fetcher.get_span(
                    session=session,
                    node_id=node_id,
                    expand="paragraph",
                )
            )

    sources_map: dict[str, SourceItem] = {}
    for item in evidence[:5]:
        sources_map.setdefault(
            item.source_label,
            SourceItem(
                source_label=item.source_label,
                page_start=item.page_start,
                page_end=item.page_end,
            ),
        )
    for hit in exact_phrase_hits[:5]:
        source_label = str(hit["source_label"])
        sources_map.setdefault(
            source_label,
            SourceItem(
                source_label=source_label,
                page_start=int(hit["page_start"]),
                page_end=int(hit["page_end"]),
            ),
        )

    answer = await answering_service.synthesize_answer(
        question=payload.question,
        intent=plan.intent,
        evidence=evidence,
        exact_phrase_hits=exact_phrase_hits,
    )
    debug = None
    if payload.include_debug:
        debug = {
            "intent": plan.intent,
            "queries": [query.model_dump() for query in plan.queries],
            "exact_phrase_hits": exact_phrase_hits,
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
