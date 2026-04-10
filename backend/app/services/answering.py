"""Facade service for query planning, evidence assessment, and answer synthesis."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.services.answering_components import (
    QuestionStructureAnalyzer,
    QuestionStructureParser,
    ToolAgentResult,
    ToolDrivenAnsweringAgent,
)
from app.services.openai_client import get_openai_client
from app.services.retrieval_components import BM25Service, DenseRetriever, HybridRetriever, StructuralContentRetriever


class AnsweringService:
    """Facade over the tool-driven answering agent."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = get_openai_client()
        self.structure_parser = QuestionStructureParser()
        self.structure_analyzer = QuestionStructureAnalyzer(
            settings=self.settings,
            client=self.client,
            parser=self.structure_parser,
        )
        self.tool_agent = ToolDrivenAnsweringAgent(
            settings=self.settings,
            client=self.client,
            structure_analyzer=self.structure_analyzer,
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
    ) -> ToolAgentResult:
        """Answer a question through LLM-selected retrieval tools."""

        return await self.tool_agent.answer_question(
            question=question,
            session=session,
            bm25_service=bm25_service,
            dense_retriever=dense_retriever,
            hybrid_retriever=hybrid_retriever,
            structural_retriever=structural_retriever,
        )
