"""Evidence sufficiency assessment for the HIPAA answering pipeline."""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI
from pydantic import ValidationError

from app.config import Settings
from app.schemas import EvidenceDecision, QueryVariant, RetrievalEvidence, StructuralFilters
from app.services.answering_components.common import (
    build_evidence_payload,
    sanitize_evidence_decision,
    sanitize_query_variants,
)
from app.services.answering_components.heuristics import StructuralQueryInterpreter
from app.services.answering_components.prompts import EVIDENCE_SYSTEM_PROMPT, build_evidence_prompt
from app.services.text_utils import unique_preserve_order

logger = logging.getLogger(__name__)


class EvidenceAssessor:
    """Judge whether current retrieval evidence is enough to answer the question."""

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

    async def assess_evidence(
        self,
        *,
        question: str,
        intent: str,
        evidence: list[RetrievalEvidence],
        attempted_queries: list[QueryVariant],
        retrieval_round: int,
    ) -> EvidenceDecision:
        """Return a sufficiency decision for the current evidence set."""

        if intent == "structure_lookup" and self.interpreter.should_return_raw_structure(question) and any(
            item.retrieval_mode == "structure_lookup" for item in evidence
        ):
            return EvidenceDecision(
                sufficient=True,
                rationale="Direct structural content was found for the request.",
            )

        outline_sufficiency = self._outline_sufficiency_decision(question, evidence)
        if outline_sufficiency is not None:
            return outline_sufficiency

        outline_follow_up = self._outline_follow_up_decision(question, evidence)
        if outline_follow_up is not None:
            return outline_follow_up

        section_follow_up = self._section_follow_up_decision(question, evidence)
        if section_follow_up is not None:
            return section_follow_up

        fallback = self._fallback_evidence_decision(
            question=question,
            intent=intent,
            evidence=evidence,
        )
        attempted_queries = sanitize_query_variants(
            attempted_queries,
            settings=self.settings,
        )
        if not self.settings.openai_api_key:
            return fallback

        prompt = build_evidence_prompt(
            question=question,
            intent=intent,
            retrieval_round=retrieval_round,
            max_rounds=self.settings.agent_max_rounds,
            attempted_queries=attempted_queries,
            evidence_payload=build_evidence_payload(evidence, limit=8),
            query_rewrite_limit=self.settings.query_rewrite_limit,
        )
        messages = [
            {"role": "system", "content": EVIDENCE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.openai_chat_model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.choices[0].message.content or "{}")
            decision = EvidenceDecision.model_validate(payload)
            return sanitize_evidence_decision(decision, settings=self.settings)
        except (json.JSONDecodeError, ValidationError, AttributeError, TypeError) as exc:
            logger.warning("Falling back to heuristic evidence assessment: %s", exc)
            return fallback
        except Exception:
            logger.exception("LLM evidence assessment failed; using heuristic fallback.")
            return fallback

    def _fallback_evidence_decision(
        self,
        *,
        question: str,
        intent: str,
        evidence: list[RetrievalEvidence],
    ) -> EvidenceDecision:
        if intent == "existence_check":
            sufficient = len(evidence) >= 2
            rationale = (
                "Enough retrieval evidence was found to answer the mention or existence question."
                if sufficient
                else "Need broader lexical or hybrid retrieval for the mention check."
            )
            missing_information = [] if sufficient else ["Need stronger mention-level evidence."]
            return EvidenceDecision(
                sufficient=sufficient,
                rationale=rationale,
                missing_information=missing_information,
            )

        if intent == "structure_lookup" and self.interpreter.should_return_raw_structure(question):
            return EvidenceDecision(
                sufficient=bool(evidence),
                rationale=(
                    "Direct structural content was found."
                    if evidence
                    else "Need a direct section or outline lookup result."
                ),
                missing_information=[] if evidence else ["Need the requested structural content."],
            )

        sufficient = len(evidence) >= 3
        return EvidenceDecision(
            sufficient=sufficient,
            rationale="Evidence set is sufficient." if sufficient else "Need more supporting chunks.",
            missing_information=[] if sufficient else ["Need additional supporting chunks."],
        )

    def _outline_sufficiency_decision(
        self,
        question: str,
        evidence: list[RetrievalEvidence],
    ) -> EvidenceDecision | None:
        structural_outlines = [
            item
            for item in evidence
            if item.retrieval_mode == "structure_lookup"
            and item.metadata.get("content_type") in {"part_outline", "subpart_outline"}
        ]
        if not structural_outlines:
            return None

        lowered_question = question.lower()
        if not any(keyword in lowered_question for keyword in ("purpose", "overview", "cover", "covers", "about")):
            return None

        for item in structural_outlines:
            matching_sections = self.interpreter.best_outline_sections(question, item)
            if matching_sections:
                section_labels = ", ".join(section["section"] for section in matching_sections[:2])
                return EvidenceDecision(
                    sufficient=True,
                    rationale=(
                        "The structural outline is sufficient for a high-level answer because the section titles "
                        f"directly indicate the relevant topic, including {section_labels}."
                    ),
                )
        return None

    def _outline_follow_up_decision(
        self,
        question: str,
        evidence: list[RetrievalEvidence],
    ) -> EvidenceDecision | None:
        structural_outlines = [
            item
            for item in evidence
            if item.retrieval_mode == "structure_lookup"
            and item.metadata.get("content_type") in {"part_outline", "subpart_outline"}
        ]
        if not structural_outlines:
            return None

        next_queries: list[QueryVariant] = []
        missing_information: list[str] = []
        for item in structural_outlines:
            for section in self.interpreter.best_outline_sections(question, item)[:2]:
                section_number = section.get("section_number")
                if not section_number:
                    continue
                next_queries.append(
                    QueryVariant(
                        text=f"{section_number} {section.get('section_title') or ''}".strip(),
                        mode="structure_lookup",
                        structure_target="section_text",
                        strategy="outline_to_section_text",
                        reason="The outline suggests this section is most likely to contain the answer.",
                        filters=StructuralFilters(section_number=section_number),
                    )
                )
                missing_information.append(
                    f"Need the full text of § {section_number} to answer more precisely."
                )

        next_queries = sanitize_query_variants(next_queries, settings=self.settings)
        if not next_queries:
            return None
        return EvidenceDecision(
            sufficient=False,
            rationale="The outline identified likely relevant sections, but the full section text is needed.",
            missing_information=unique_preserve_order(missing_information),
            next_queries=next_queries,
        )

    def _section_follow_up_decision(
        self,
        question: str,
        evidence: list[RetrievalEvidence],
    ) -> EvidenceDecision | None:
        existing_full_sections = {
            item.metadata.get("section_number")
            for item in evidence
            if item.retrieval_mode == "structure_lookup"
            and item.metadata.get("content_type") == "section_text"
            and item.metadata.get("section_number")
        }

        ranked_sections = self.interpreter.best_referenced_sections(question, evidence)
        next_queries: list[QueryVariant] = []
        missing_information: list[str] = []
        for section in ranked_sections[:2]:
            section_number = str(section.get("section_number") or "").strip()
            if not section_number or section_number in existing_full_sections:
                continue
            section_title = str(section.get("section_title") or "").strip()
            next_queries.append(
                QueryVariant(
                    text=f"{section_number} {section_title}".strip(),
                    mode="structure_lookup",
                    structure_target="section_text",
                    strategy="section_chunk_to_full_text",
                    reason="Retrieved chunks suggest this section is central, so fetch the full section text.",
                    filters=StructuralFilters(section_number=section_number),
                )
            )
            missing_information.append(
                f"Need the full text of § {section_number} to answer the question precisely."
            )

        next_queries = sanitize_query_variants(next_queries, settings=self.settings)
        if not next_queries:
            return None

        return EvidenceDecision(
            sufficient=False,
            rationale="Partial chunks point to specific relevant sections, but the full section text is still needed.",
            missing_information=unique_preserve_order(missing_information),
            next_queries=next_queries,
        )
