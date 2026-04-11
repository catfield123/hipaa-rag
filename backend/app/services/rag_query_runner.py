"""Use case: run one RAG chat query and build the public API response."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.chat import ChatQueryResponse
from app.services.answering import (
    AgentAnswerDeltaEmitter,
    AgentStatusEmitter,
    AnsweringService,
)
from app.services.rag_response_builder import RagResponseBuilder
from app.services.retrieval_components import (
    BM25Service,
    DenseRetriever,
    HybridRetriever,
    StructuralContentRetriever,
)


async def run_rag_query(
    *,
    question: str,
    session: AsyncSession,
    answering_service: AnsweringService,
    rag_response_builder: RagResponseBuilder,
    bm25_service: BM25Service,
    dense_retriever: DenseRetriever,
    hybrid_retriever: HybridRetriever,
    structural_retriever: StructuralContentRetriever,
    on_status: AgentStatusEmitter | None = None,
    on_answer_delta: AgentAnswerDeltaEmitter | None = None,
) -> ChatQueryResponse:
    """Execute retrieval and answering, then assemble ``ChatQueryResponse``.

    Args:
        question (str): User question text.
        session (AsyncSession): Database session for retrieval.
        answering_service (AnsweringService): Function-calling answering pipeline.
        rag_response_builder (RagResponseBuilder): Maps evidence to quotes and sources.
        bm25_service (BM25Service): Lexical retrieval backend.
        dense_retriever (DenseRetriever): Dense vector retrieval backend.
        hybrid_retriever (HybridRetriever): Hybrid retrieval backend.
        structural_retriever (StructuralContentRetriever): Structural lookup backend.
        on_status (AgentStatusEmitter | None): Optional async callback for status events.
        on_answer_delta (AgentAnswerDeltaEmitter | None): Optional streaming answer chunks.

    Returns:
        ChatQueryResponse: Answer text, quotes, sources, intent, and round count.

    Raises:
        ConfigurationError: When required configuration (e.g. OpenAI API key) is missing.
        RuntimeError: When the agent cannot complete a retrieval round as required.
    """

    outcome = await answering_service.answer_question(
        question=question,
        session=session,
        bm25_service=bm25_service,
        dense_retriever=dense_retriever,
        hybrid_retriever=hybrid_retriever,
        structural_retriever=structural_retriever,
        on_status=on_status,
        on_answer_delta=on_answer_delta,
    )
    return ChatQueryResponse(
        answer=outcome.answer,
        quotes=rag_response_builder.build_quotes(outcome.evidence),
        sources=rag_response_builder.build_sources(outcome.evidence),
        intent=outcome.intent,
        retrieval_rounds=outcome.retrieval_rounds,
    )
