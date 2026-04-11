"""RAG API routes."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.api.deps import (
    AnsweringServiceDep,
    Bm25ServiceDep,
    DbSessionDep,
    DenseRetrieverDep,
    HybridRetrieverDep,
    RagResponseBuilderDep,
    StructuralRetrieverDep,
)
from app.core.exceptions import AppError
from app.schemas.chat import ChatQueryRequest, ChatQueryResponse
from app.schemas.ws_events import (
    WsAnswerDeltaEvent,
    WsErrorEvent,
    WsStatusEvent,
    ws_result_event_payload,
)
from app.services.rag_query_runner import run_rag_query
from fastapi import APIRouter, WebSocket
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/rag",
    tags=["rag"],
)


@router.post(
    "/query",
    response_model=ChatQueryResponse,
    summary="Run RAG query",
    description=(
        "Runs the retrieval and function-calling agent, then returns the answer with "
        "quotes, sources, inferred intent, and retrieval round count."
    ),
)
async def query_rag(
    payload: ChatQueryRequest,
    session: DbSessionDep,
    answering_service: AnsweringServiceDep,
    rag_response_builder: RagResponseBuilderDep,
    bm25_service: Bm25ServiceDep,
    dense_retriever: DenseRetrieverDep,
    hybrid_retriever: HybridRetrieverDep,
    structural_retriever: StructuralRetrieverDep,
) -> ChatQueryResponse:
    """Run the retrieval loop and return the RAG response payload.

    Args:
        payload (ChatQueryRequest): Validated request body with the user question.
        session (AsyncSession): Request-scoped database session.
        answering_service (AnsweringService): Agent service.
        rag_response_builder (RagResponseBuilder): Maps evidence to client quote/source lists.
        bm25_service (BM25Service): BM25 backend.
        dense_retriever (DenseRetriever): Dense retrieval backend.
        hybrid_retriever (HybridRetriever): Hybrid retrieval backend.
        structural_retriever (StructuralContentRetriever): Structural lookup backend.

    Returns:
        ChatQueryResponse: Answer, quotes, sources, intent, and retrieval rounds.

    Raises:
        ConfigurationError: If required API configuration is missing.
        RuntimeError: If the agent cannot complete a retrieval round as required.
    """

    return await run_rag_query(
        question=payload.question,
        session=session,
        answering_service=answering_service,
        rag_response_builder=rag_response_builder,
        bm25_service=bm25_service,
        dense_retriever=dense_retriever,
        hybrid_retriever=hybrid_retriever,
        structural_retriever=structural_retriever,
    )


@router.websocket("/query/ws")
async def query_rag_ws(
    websocket: WebSocket,
    session: DbSessionDep,
    answering_service: AnsweringServiceDep,
    rag_response_builder: RagResponseBuilderDep,
    bm25_service: Bm25ServiceDep,
    dense_retriever: DenseRetrieverDep,
    hybrid_retriever: HybridRetrieverDep,
    structural_retriever: StructuralRetrieverDep,
) -> None:
    """Stream agent status and answer deltas, then send the same payload as ``POST /rag/query``.

    The client must send a single text frame containing JSON: ``{\"question\": \"...\"}``
    (same shape as :class:`ChatQueryRequest`).

    Args:
        websocket (WebSocket): Accepted browser/client connection.
        session (AsyncSession): Database session for the lifetime of the handler.
        answering_service (AnsweringService): Shared answering service.
        rag_response_builder (RagResponseBuilder): Response assembler.
        bm25_service (BM25Service): BM25 backend.
        dense_retriever (DenseRetriever): Dense retrieval backend.
        hybrid_retriever (HybridRetriever): Hybrid retrieval backend.
        structural_retriever (StructuralContentRetriever): Structural lookup backend.

    Returns:
        None: Connection is closed after a ``result`` or ``error`` message.

    Raises:
        None: Errors are sent as ``type: error`` JSON and the socket is closed with an appropriate code.
    """

    await websocket.accept()
    try:
        raw = await websocket.receive_text()
    except WebSocketDisconnect:
        return

    try:
        payload_obj = json.loads(raw)
    except json.JSONDecodeError:
        err = WsErrorEvent(message="Expected JSON with a question field.")
        await websocket.send_json(err.model_dump(mode="json"))
        await websocket.close(code=1007)
        return

    try:
        request = ChatQueryRequest.model_validate(payload_obj)
    except ValidationError as exc:
        err = WsErrorEvent(message=str(exc))
        await websocket.send_json(err.model_dump(mode="json"))
        await websocket.close(code=1007)
        return

    async def on_status(event: dict[str, Any]) -> None:
        validated = WsStatusEvent.model_validate(event)
        await websocket.send_json(validated.model_dump(mode="json", by_alias=True))

    async def on_answer_delta(text: str) -> None:
        msg = WsAnswerDeltaEvent(text=text)
        await websocket.send_json(msg.model_dump(mode="json"))

    try:
        response = await run_rag_query(
            question=request.question,
            session=session,
            answering_service=answering_service,
            rag_response_builder=rag_response_builder,
            bm25_service=bm25_service,
            dense_retriever=dense_retriever,
            hybrid_retriever=hybrid_retriever,
            structural_retriever=structural_retriever,
            on_status=on_status,
            on_answer_delta=on_answer_delta,
        )
    except AppError as exc:
        logger.warning("RAG WebSocket app error: %s", exc.message)
        err = WsErrorEvent(message=exc.message)
        await websocket.send_json(err.model_dump(mode="json"))
        await websocket.close(code=exc.ws_close_code)
        return
    except Exception:
        logger.exception("RAG WebSocket internal error")
        err = WsErrorEvent(message="An internal error occurred.")
        await websocket.send_json(err.model_dump(mode="json"))
        await websocket.close(code=1011)
        return

    await websocket.send_json(ws_result_event_payload(response))
    await websocket.close(code=1000)
