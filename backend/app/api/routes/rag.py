"""RAG API routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db_session
from app.schemas.chat import ChatQueryRequest, ChatQueryResponse
from app.services.answering import AnsweringService
from app.services.dependencies import get_answering_service, get_rag_response_builder
from app.services.rag_response_builder import RagResponseBuilder
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

router = APIRouter(prefix="/rag", tags=["rag"])
DbSessionDep = Annotated[AsyncSession, Depends(get_db_session)]
Bm25ServiceDep = Annotated[BM25Service, Depends(get_bm25_service)]
DenseRetrieverDep = Annotated[DenseRetriever, Depends(get_dense_retriever)]
HybridRetrieverDep = Annotated[HybridRetriever, Depends(get_hybrid_retriever)]
StructuralRetrieverDep = Annotated[
    StructuralContentRetriever,
    Depends(get_structural_content_retriever),
]


AnsweringServiceDep = Annotated[AnsweringService, Depends(get_answering_service)]
RagResponseBuilderDep = Annotated[RagResponseBuilder, Depends(get_rag_response_builder)]


@router.post("/query")
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
    """Run the retrieval loop and return the RAG response payload."""

    outcome = await answering_service.answer_question(
        question=payload.question,
        session=session,
        bm25_service=bm25_service,
        dense_retriever=dense_retriever,
        hybrid_retriever=hybrid_retriever,
        structural_retriever=structural_retriever,
    )

    return ChatQueryResponse(
        answer=outcome.answer,
        quotes=rag_response_builder.build_quotes(outcome.evidence),
        sources=rag_response_builder.build_sources(outcome.evidence),
        intent=outcome.intent,
        retrieval_rounds=outcome.retrieval_rounds,
    )
