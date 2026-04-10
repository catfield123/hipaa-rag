"""Structure-aware parsing and LLM-assisted interpretation helpers."""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.schemas import QueryIntent, StructuralFilters

logger = logging.getLogger(__name__)

STRUCTURE_ANALYSIS_SYSTEM_PROMPT = (
    "You analyze questions about HIPAA structure. Return JSON only. "
    "Decide whether the user wants raw structural content such as the full text of a section, "
    "a subpart outline, or a part outline, or whether they want a synthesized answer. "
    "Also infer the best intent hint for downstream retrieval planning."
)


class QuestionStructureAnalysis(BaseModel):
    """Structured interpretation of whether a question asks for raw structure."""

    intent_hint: QueryIntent
    wants_raw_structure: bool
    rationale: str


class QuestionStructureParser:
    """Extract explicit structural references without encoding answer logic."""

    def infer_structural_filters(self, user_query: str) -> StructuralFilters | None:
        """Extract cited part, section, subpart, and marker references from a question."""

        lowered = user_query.lower()
        part_match = re.search(r"\bpart\s+(\d{3})\b", lowered, flags=re.IGNORECASE)
        section_match = re.search(r"(?:§\s*)?(\d{3}\.\d+)", lowered, flags=re.IGNORECASE)
        subpart_match = re.search(r"\bsubpart\s+([a-z])\b", lowered, flags=re.IGNORECASE)
        marker_path = re.findall(r"\(([A-Za-z0-9ivxlcdmIVXLCDM]+)\)", user_query)

        if not (part_match or section_match or subpart_match or marker_path):
            return None

        return StructuralFilters(
            part_number=part_match.group(1) if part_match else None,
            section_number=section_match.group(1) if section_match else None,
            subpart=subpart_match.group(1).upper() if subpart_match else None,
            marker_path=[marker.lower() for marker in marker_path],
        )


class QuestionStructureAnalyzer:
    """Use the LLM to interpret whether a question requests raw structure."""

    def __init__(
        self,
        *,
        settings: Settings,
        client: AsyncOpenAI,
        parser: QuestionStructureParser,
    ) -> None:
        self.settings = settings
        self.client = client
        self.parser = parser
        self._cache: dict[str, QuestionStructureAnalysis] = {}

    async def analyze_question(self, question: str) -> QuestionStructureAnalysis:
        """Return a structured interpretation of the question's structural intent."""

        cached = self._cache.get(question)
        if cached is not None:
            return cached

        fallback = self._fallback_analysis(question)
        if not self.settings.openai_api_key:
            self._cache[question] = fallback
            return fallback

        payload = {
            "question": question,
            "explicit_structural_references": (
                self.parser.infer_structural_filters(question).model_dump()
                if self.parser.infer_structural_filters(question)
                else None
            ),
            "instructions": [
                "Set wants_raw_structure=true only when the user explicitly wants full text, an outline, "
                "a list of sections, or direct display of structure.",
                "Set wants_raw_structure=false for overview, purpose, explanation, scope, or analytical questions, "
                "even if they mention a part, subpart, or section.",
                "Choose intent_hint from: general, quote_request, existence_check, structure_lookup.",
            ],
        }
        messages = [
            {"role": "system", "content": STRUCTURE_ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ]
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.openai_chat_model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            raw_payload = json.loads(response.choices[0].message.content or "{}")
            analysis = QuestionStructureAnalysis.model_validate(raw_payload)
            self._cache[question] = analysis
            return analysis
        except (json.JSONDecodeError, ValidationError, AttributeError, TypeError) as exc:
            logger.warning("Falling back to simple structure analysis: %s", exc)
            self._cache[question] = fallback
            return fallback
        except Exception:
            logger.exception("LLM structure analysis failed; using fallback.")
            self._cache[question] = fallback
            return fallback

    def cached_or_fallback(self, question: str) -> QuestionStructureAnalysis:
        """Return cached structure analysis or a lightweight fallback analysis."""

        return self._cache.get(question) or self._fallback_analysis(question)

    def _fallback_analysis(self, question: str) -> QuestionStructureAnalysis:
        lowered = question.lower()
        filters = self.parser.infer_structural_filters(question)
        has_direct_display_signal = any(
            phrase in lowered
            for phrase in (
                "full text",
                "entire section",
                "whole section",
                "text of",
                "show",
                "display",
                "outline",
                "list",
                "all sections",
                "section headings",
            )
        )
        wants_raw_structure = filters is not None and has_direct_display_signal
        if wants_raw_structure:
            intent_hint: QueryIntent = "structure_lookup"
        elif any(word in lowered for word in ("does", "mention", "is there")):
            intent_hint = "existence_check"
        elif any(word in lowered for word in ("quote", "cite", "specific regulation")):
            intent_hint = "quote_request"
        else:
            intent_hint = "general"
        return QuestionStructureAnalysis(
            intent_hint=intent_hint,
            wants_raw_structure=wants_raw_structure,
            rationale="Fallback structure analysis based on explicit references and display signals.",
        )
