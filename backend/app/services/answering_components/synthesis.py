"""Answer synthesis and rendering for the HIPAA answering pipeline."""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

from app.config import Settings
from app.schemas import RetrievalEvidence
from app.services.answering_components.heuristics import StructuralQueryInterpreter
from app.services.answering_components.prompts import ANSWER_SYSTEM_PROMPT, build_answer_prompt
from app.services.text_utils import unique_preserve_order

logger = logging.getLogger(__name__)


class AnswerSynthesizer:
    """Turn supported evidence into user-facing answers."""

    def __init__(
        self,
        *,
        settings: Settings,
        client: AsyncOpenAI,
        interpreter: StructuralQueryInterpreter,
    ) -> None:
        self.settings = settings
        self.client = client
        self.interpreter = interpreter

    async def synthesize_answer(
        self,
        *,
        question: str,
        intent: str,
        evidence: list[RetrievalEvidence],
    ) -> str:
        """Generate a final answer using only the provided evidence."""

        if not evidence:
            return "I could not find enough support in the provided HIPAA text to answer confidently."

        if intent == "structure_lookup" and self.interpreter.should_return_raw_structure(question):
            return self.render_structural_content(evidence)

        if not self.settings.openai_api_key:
            return self.fallback_answer(question=question, intent=intent, evidence=evidence)

        prompt = build_answer_prompt(
            question=question,
            intent=intent,
            evidence=evidence,
        )
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.openai_chat_model,
                messages=messages,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            logger.exception("LLM answer synthesis failed; using fallback answer.")
            return self.fallback_answer(question=question, intent=intent, evidence=evidence)

    def render_insufficient_answer(
        self,
        *,
        evidence: list[RetrievalEvidence],
        rationale: str,
        retrieval_rounds: int,
    ) -> str:
        """Render a user-facing answer for insufficient evidence cases."""

        lead = (
            f"I could not find enough support in the provided HIPAA text after {retrieval_rounds} retrieval rounds."
        )
        if not evidence:
            return f"{lead} {rationale}".strip()

        sources = ", ".join(unique_preserve_order(item.path_text for item in evidence[:3]))
        return f"{lead} Closest supporting passages: {sources}. {rationale}".strip()

    def fallback_answer(
        self,
        *,
        question: str,
        intent: str,
        evidence: list[RetrievalEvidence],
    ) -> str:
        """Return a deterministic fallback answer when LLM synthesis is unavailable."""

        if intent == "structure_lookup" and self.interpreter.should_return_raw_structure(question):
            return self.render_structural_content(evidence)

        if intent == "existence_check":
            if evidence:
                sources = ", ".join(item.path_text for item in evidence[:3])
                return (
                    "I found HIPAA passages relevant to that mention or existence question. "
                    f"Strongest sources: {sources}."
                )
            return "I did not find enough BM25 or hybrid evidence for that wording in the provided HIPAA text."

        lead = evidence[0]
        return (
            f"Based on the retrieved HIPAA text, the strongest supporting source is {lead.path_text}. "
            f"Relevant excerpt: {lead.text}"
        )

    def render_structural_content(self, evidence: list[RetrievalEvidence]) -> str:
        """Return raw structural content when the user explicitly asked for it."""

        structural_texts = [
            item.text.strip()
            for item in evidence
            if item.retrieval_mode == "structure_lookup" and item.text.strip()
        ]
        if structural_texts:
            return "\n\n".join(unique_preserve_order(structural_texts))
        return self.fallback_answer(question="", intent="general", evidence=evidence)
