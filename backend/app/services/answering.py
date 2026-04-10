"""Facade service over the function-calling answering agent."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.services.answering_components import (
    FunctionAgentResult,
    FunctionCallingAnsweringAgent,
    QuestionStructureParser,
)
from app.services.openai_client import get_openai_client
from app.services.retrieval_components import BM25Service, DenseRetriever, HybridRetriever, StructuralContentRetriever


class AnsweringService:
    """Facade over the function-calling answering agent."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = get_openai_client()
        self.structure_parser = QuestionStructureParser()
        self.function_agent = FunctionCallingAnsweringAgent(
            settings=self.settings,
            client=self.client,
            structure_parser=self.structure_parser,
        )

    async def answer_question(
        self,
        *,
        question: str,
        session: AsyncSession,
        bm25_service: BM25Service,
        dense_retriever: DenseRetriever,
        hybrid_retriever: HybridRetriever,
        structural_retriever: StructuralContentRetriever,
    ) -> FunctionAgentResult:
        """Answer a question through LLM-selected retrieval functions."""

        return await self.function_agent.answer_question(
            question=question,
            session=session,
            bm25_service=bm25_service,
            dense_retriever=dense_retriever,
            hybrid_retriever=hybrid_retriever,
            structural_retriever=structural_retriever,
        )
