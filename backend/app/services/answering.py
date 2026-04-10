from __future__ import annotations

import json
import re

from app.config import get_settings
from app.schemas import EvidenceDecision, QueryPlan, QueryVariant, RetrievalEvidence, StructuralFilters
from app.services.openai_client import get_openai_client
from app.services.text_utils import tokenize, unique_preserve_order


class AnsweringService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = get_openai_client()

    async def plan_queries(
        self,
        user_query: str,
        retrieval_round: int = 1,
        previous_failed_queries: list[str] | None = None,
        intent_hint: str | None = None,
    ) -> QueryPlan:
        fallback = self._fallback_query_plan(user_query)
        if not self.settings.openai_api_key:
            return fallback

        prompt = {
            "user_query": user_query,
            "retrieval_round": retrieval_round,
            "previous_failed_queries": previous_failed_queries or [],
            "intent_hint": intent_hint,
            "rules": {
                "return_between_1_and_6_queries": True,
                "allowed_modes": ["bm25_only", "hybrid"],
                "bm25_is_required_for_literal_or_mention_queries": True,
                "drop_corpus_obvious_terms_like_hipaa": True,
                "optional_structural_filters": {
                    "part_number": "string | null",
                    "section_number": "string | null",
                    "subpart": "string | null",
                    "marker_path": ["string"],
                },
            },
        }

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a legal retrieval planner. Return JSON only. "
                    "Rewrite the user query into retrieval queries. "
                    "Choose bm25_only for literal or mention checks. "
                    "Choose hybrid for semantic or multi-section questions. "
                    "If the user asks about a specific part/section/subpart/paragraph, "
                    "add filters to narrow retrieval."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt),
            },
        ]
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.openai_chat_model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.choices[0].message.content or "{}")
            return QueryPlan.model_validate(payload)
        except Exception:
            return fallback

    async def judge_evidence(
        self,
        question: str,
        intent: str,
        evidence: list[RetrievalEvidence],
    ) -> EvidenceDecision:
        if intent == "existence_check":
            sufficient = len(evidence) >= 2
            rationale = (
                "Enough retrieval evidence was found to answer the mention/existence question."
                if sufficient
                else "Need broader lexical or hybrid retrieval for the mention check."
            )
            actions = [] if sufficient else ["broaden bm25", "hybrid follow-up"]
            return EvidenceDecision(sufficient=sufficient, rationale=rationale, follow_up_actions=actions)

        sufficient = len(evidence) >= 3
        return EvidenceDecision(
            sufficient=sufficient,
            rationale="Evidence set is sufficient." if sufficient else "Need more supporting chunks.",
            follow_up_actions=[] if sufficient else ["hybrid follow-up"],
        )

    async def synthesize_answer(
        self,
        question: str,
        intent: str,
        evidence: list[RetrievalEvidence],
    ) -> str:
        if not evidence:
            return "I could not find enough support in the provided HIPAA text to answer confidently."

        if not self.settings.openai_api_key:
            return self._fallback_answer(question, intent, evidence)

        evidence_payload = [
            {
                "path_text": item.path_text,
                "part": item.part,
                "subpart": item.subpart,
                "section": item.section,
                "markers": item.markers,
                "text": item.text,
            }
            for item in evidence[:6]
        ]
        prompt = {
            "question": question,
            "intent": intent,
            "evidence": evidence_payload,
            "instructions": {
                "answer_only_from_evidence": True,
                "say_not_found_if_insufficient": True,
                "be_explicit_for_negative_questions": True,
            },
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You answer questions about HIPAA using only provided evidence. "
                    "Do not hallucinate. If the question is an existence check, "
                    "answer only from the retrieved BM25 or hybrid evidence. "
                    "Use the structural references in each chunk when citing support."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt),
            },
        ]
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.openai_chat_model,
                messages=messages,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            return self._fallback_answer(question, intent, evidence)

    def _fallback_query_plan(self, user_query: str) -> QueryPlan:
        cleaned = re.sub(r"\bhipaa\b", "", user_query, flags=re.IGNORECASE).strip(" ?")
        keywords = tokenize(cleaned) or tokenize(user_query)
        exact_query = cleaned if cleaned else user_query.strip()
        broad_query = " ".join(unique_preserve_order(keywords[:4])) or exact_query
        inferred_filters = self._infer_structural_filters(user_query)

        lowered = user_query.lower()
        if any(word in lowered for word in ("does", "mention", "is there", "does hipaa")):
            intent = "existence_check"
            queries = [
                QueryVariant(
                    text=exact_query,
                    mode="bm25_only",
                    strategy="bm25_literal",
                    reason="Use lexical retrieval for mention or existence checks.",
                    filters=inferred_filters,
                ),
                QueryVariant(
                    text=broad_query,
                    mode="hybrid",
                    strategy="hybrid_support",
                    reason="Broaden recall with semantic support around the lexical anchor.",
                    filters=inferred_filters,
                ),
            ]
            return QueryPlan(intent=intent, queries=queries)

        if any(word in lowered for word in ("quote", "cite", "specific regulation", "full text")):
            intent = "quote_request"
            queries = [
                QueryVariant(
                    text=exact_query,
                    mode="bm25_only",
                    strategy="citation_literal",
                    reason="Literal regulation retrieval for quotes.",
                    filters=inferred_filters,
                ),
                QueryVariant(
                    text=broad_query,
                    mode="hybrid",
                    strategy="semantic_support",
                    reason="Add surrounding semantic context.",
                    filters=inferred_filters,
                ),
            ]
            return QueryPlan(intent=intent, queries=queries)

        queries = [
            QueryVariant(
                text=exact_query,
                mode="bm25_only",
                strategy="lexical_anchor",
                reason="Anchor the query with literal regulatory terms.",
                filters=inferred_filters,
            ),
            QueryVariant(
                text=broad_query,
                mode="hybrid",
                strategy="semantic_expansion",
                reason="Retrieve broader context and related sections.",
                filters=inferred_filters,
            ),
        ]
        return QueryPlan(intent="general", queries=queries)

    def _infer_structural_filters(self, user_query: str) -> StructuralFilters | None:
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

    def _fallback_answer(
        self,
        question: str,
        intent: str,
        evidence: list[RetrievalEvidence],
    ) -> str:
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
