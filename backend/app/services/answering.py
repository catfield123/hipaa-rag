from __future__ import annotations

import json
import re
from typing import Any

from app.config import get_settings
from app.schemas import EvidenceDecision, QueryPlan, QueryVariant, RetrievalEvidence
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
            },
        }

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a legal retrieval planner. Return JSON only. "
                    "Rewrite the user query into retrieval queries. "
                    "Choose bm25_only for literal or mention checks. "
                    "Choose hybrid for semantic or multi-section questions."
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
        exact_phrase_hits: list[dict[str, Any]],
    ) -> EvidenceDecision:
        if exact_phrase_hits:
            return EvidenceDecision(
                sufficient=True,
                rationale="Exact phrase evidence was found.",
                follow_up_actions=[],
            )

        if intent == "existence_check":
            sufficient = len(evidence) >= 2
            rationale = (
                "Enough evidence to distinguish exact wording from related concepts."
                if sufficient
                else "Need broader or more literal retrieval for mention check."
            )
            actions = [] if sufficient else ["exact phrase lookup", "broaden bm25"]
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
        exact_phrase_hits: list[dict[str, Any]],
    ) -> str:
        if not evidence and not exact_phrase_hits:
            return "I could not find enough support in the provided HIPAA text to answer confidently."

        if not self.settings.openai_api_key:
            return self._fallback_answer(question, intent, evidence, exact_phrase_hits)

        evidence_payload = [
            {
                "source_label": item.source_label,
                "pages": [item.page_start, item.page_end],
                "content": item.content,
            }
            for item in evidence[:6]
        ]
        prompt = {
            "question": question,
            "intent": intent,
            "exact_phrase_hits": exact_phrase_hits,
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
                    "clearly separate exact wording from related concepts."
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
            return self._fallback_answer(question, intent, evidence, exact_phrase_hits)

    def _fallback_query_plan(self, user_query: str) -> QueryPlan:
        cleaned = re.sub(r"\bhipaa\b", "", user_query, flags=re.IGNORECASE).strip(" ?")
        keywords = tokenize(cleaned) or tokenize(user_query)
        exact_query = cleaned if cleaned else user_query.strip()
        broad_query = " ".join(unique_preserve_order(keywords[:4])) or exact_query
        semantic_query = broad_query

        lowered = user_query.lower()
        if any(word in lowered for word in ("does", "mention", "is there", "does hipaa")):
            intent = "existence_check"
            queries = [
                QueryVariant(
                    text=exact_query,
                    mode="bm25_only",
                    strategy="bm25_exact",
                    reason="Check exact wording directly in the corpus.",
                ),
                QueryVariant(
                    text=broad_query,
                    mode="bm25_only",
                    strategy="bm25_broad",
                    reason="Check whether related literal terminology appears.",
                ),
                QueryVariant(
                    text=semantic_query,
                    mode="hybrid",
                    strategy="semantic_probe",
                    reason="Look for nearby concepts if wording differs.",
                ),
            ]
            return QueryPlan(intent=intent, needs_exact_phrase_check=True, queries=queries)

        if any(word in lowered for word in ("quote", "cite", "specific regulation", "full text")):
            intent = "quote_request"
            queries = [
                QueryVariant(
                    text=exact_query,
                    mode="bm25_only",
                    strategy="citation_literal",
                    reason="Literal regulation retrieval for quotes.",
                ),
                QueryVariant(
                    text=broad_query,
                    mode="hybrid",
                    strategy="semantic_support",
                    reason="Add surrounding semantic context.",
                ),
            ]
            return QueryPlan(intent=intent, queries=queries)

        queries = [
            QueryVariant(
                text=exact_query,
                mode="bm25_only",
                strategy="lexical_anchor",
                reason="Anchor the query with literal regulatory terms.",
            ),
            QueryVariant(
                text=broad_query,
                mode="hybrid",
                strategy="semantic_expansion",
                reason="Retrieve broader context and related sections.",
            ),
        ]
        return QueryPlan(intent="general", queries=queries)

    def _fallback_answer(
        self,
        question: str,
        intent: str,
        evidence: list[RetrievalEvidence],
        exact_phrase_hits: list[dict[str, Any]],
    ) -> str:
        if intent == "existence_check":
            if exact_phrase_hits:
                sources = ", ".join(hit["source_label"] for hit in exact_phrase_hits[:3])
                return f"The document appears to mention the exact phrase or a direct match in {sources}."
            if evidence:
                sources = ", ".join(item.source_label for item in evidence[:3])
                return (
                    "I found related HIPAA passages, but I did not find a confirmed exact phrase match. "
                    f"Closest sources: {sources}."
                )
            return "I did not find support for that wording in the provided HIPAA text."

        lead = evidence[0]
        return (
            f"Based on the retrieved HIPAA text, the strongest supporting source is {lead.source_label}. "
            f"Relevant excerpt: {lead.content}"
        )
