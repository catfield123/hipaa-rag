"""Facade service for query planning, evidence assessment, and answer synthesis."""

from __future__ import annotations

from app.config import get_settings
from app.schemas import EvidenceDecision, QueryPlan, QueryVariant, RetrievalEvidence
from app.services.answering_components import (
    AnswerSynthesizer,
    EvidenceAssessor,
    QueryPlanner,
    StructuralQueryInterpreter,
)
from app.services.openai_client import get_openai_client


class AnsweringService:
    """Thin facade over the decomposed answering pipeline components."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = get_openai_client()
        self.interpreter = StructuralQueryInterpreter()
        self.planner = QueryPlanner(
            settings=self.settings,
            client=self.client,
            interpreter=self.interpreter,
        )
        self.assessor = EvidenceAssessor(
            settings=self.settings,
            client=self.client,
            interpreter=self.interpreter,
        )
        self.synthesizer = AnswerSynthesizer(
            settings=self.settings,
            client=self.client,
            interpreter=self.interpreter,
        )

    async def plan_queries(
        self,
        user_query: str,
        retrieval_round: int = 1,
        previous_failed_queries: list[str] | None = None,
        intent_hint: str | None = None,
    ) -> QueryPlan:
        """Plan retrieval queries for the incoming question."""

        return await self.planner.plan_queries(
            user_query,
            retrieval_round=retrieval_round,
            previous_failed_queries=previous_failed_queries,
            intent_hint=intent_hint,
        )

    async def assess_evidence(
        self,
        question: str,
        intent: str,
        evidence: list[RetrievalEvidence],
        attempted_queries: list[QueryVariant],
        retrieval_round: int,
    ) -> EvidenceDecision:
        """Judge whether the current evidence set is sufficient."""

        return await self.assessor.assess_evidence(
            question=question,
            intent=intent,
            evidence=evidence,
            attempted_queries=attempted_queries,
            retrieval_round=retrieval_round,
        )

    async def synthesize_answer(
        self,
        question: str,
        intent: str,
        evidence: list[RetrievalEvidence],
    ) -> str:
        """Synthesize a final answer from retrieved evidence."""

        return await self.synthesizer.synthesize_answer(
            question=question,
            intent=intent,
            evidence=evidence,
        )

    def render_insufficient_answer(
        self,
        *,
        evidence: list[RetrievalEvidence],
        rationale: str,
        retrieval_rounds: int,
    ) -> str:
        """Render a fallback answer for insufficient-evidence cases."""

        return self.synthesizer.render_insufficient_answer(
            evidence=evidence,
            rationale=rationale,
            retrieval_rounds=retrieval_rounds,
        )
