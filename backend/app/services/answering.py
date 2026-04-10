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
                "query_count_range": {
                    "min": 1,
                    "max": self.settings.query_rewrite_limit,
                },
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
            plan = QueryPlan.model_validate(payload)
            return self._sanitize_query_plan(plan) or fallback
        except Exception:
            return fallback

    async def assess_evidence(
        self,
        question: str,
        intent: str,
        evidence: list[RetrievalEvidence],
        attempted_queries: list[QueryVariant],
        retrieval_round: int,
    ) -> EvidenceDecision:
        fallback = self._fallback_evidence_decision(intent=intent, evidence=evidence)
        attempted_queries = self._sanitize_query_variants(attempted_queries)
        if not self.settings.openai_api_key:
            return fallback

        prompt = {
            "question": question,
            "intent": intent,
            "retrieval_round": retrieval_round,
            "max_rounds": self.settings.agent_max_rounds,
            "attempted_queries": [query.model_dump() for query in attempted_queries],
            "evidence": self._build_evidence_payload(evidence, limit=8),
            "rules": {
                "decide_if_the_evidence_can_answer_the_question_now": True,
                "answer_must_be_supported_by_retrieved_evidence": True,
                "if_sufficient_return_no_next_queries": True,
                "if_insufficient_return_between_1_and_n_next_queries": self.settings.query_rewrite_limit,
                "avoid_repeating_or_lightly_rephrasing_attempted_queries": True,
                "choose_bm25_only_for_literal_quote_or_exact_mention_checks": True,
                "choose_hybrid_for_semantic_follow_up": True,
                "optional_structural_filters": {
                    "part_number": "string | null",
                    "section_number": "string | null",
                    "subpart": "string | null",
                    "marker_path": ["string"],
                },
            },
            "response_schema": {
                "sufficient": "boolean",
                "rationale": "string",
                "missing_information": ["string"],
                "next_queries": [
                    {
                        "text": "string",
                        "mode": "bm25_only | hybrid",
                        "strategy": "string",
                        "reason": "string",
                        "filters": {
                            "part_number": "string | null",
                            "section_number": "string | null",
                            "subpart": "string | null",
                            "marker_path": ["string"],
                        },
                    }
                ],
            },
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a legal retrieval judge for HIPAA question answering. "
                    "Return JSON only. Decide whether the currently retrieved evidence is enough to answer "
                    "the user's question. If yes, set sufficient=true and next_queries=[]. "
                    "If no, explain what is missing and propose the next retrieval queries."
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
            decision = EvidenceDecision.model_validate(payload)
            return self._sanitize_evidence_decision(decision)
        except Exception:
            return fallback

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
            return QueryPlan(intent=intent, queries=self._sanitize_query_variants(queries))

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
            return QueryPlan(intent=intent, queries=self._sanitize_query_variants(queries))

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
        return QueryPlan(intent="general", queries=self._sanitize_query_variants(queries))

    def render_insufficient_answer(
        self,
        *,
        evidence: list[RetrievalEvidence],
        rationale: str,
        retrieval_rounds: int,
    ) -> str:
        lead = (
            f"I could not find enough support in the provided HIPAA text after {retrieval_rounds} retrieval rounds."
        )
        if not evidence:
            return f"{lead} {rationale}".strip()

        sources = ", ".join(unique_preserve_order(item.path_text for item in evidence[:3]))
        return f"{lead} Closest supporting passages: {sources}. {rationale}".strip()

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

    def _build_evidence_payload(
        self,
        evidence: list[RetrievalEvidence],
        *,
        limit: int,
    ) -> list[dict[str, object]]:
        return [
            {
                "chunk_id": item.chunk_id,
                "path_text": item.path_text,
                "part": item.part,
                "subpart": item.subpart,
                "section": item.section,
                "markers": item.markers,
                "retrieval_mode": item.retrieval_mode,
                "score": item.score,
                "text": item.text,
            }
            for item in evidence[:limit]
        ]

    def _fallback_evidence_decision(
        self,
        *,
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

        sufficient = len(evidence) >= 3
        return EvidenceDecision(
            sufficient=sufficient,
            rationale="Evidence set is sufficient." if sufficient else "Need more supporting chunks.",
            missing_information=[] if sufficient else ["Need additional supporting chunks."],
        )

    def _sanitize_query_plan(self, plan: QueryPlan) -> QueryPlan | None:
        queries = self._sanitize_query_variants(plan.queries)
        if not queries:
            return None
        return plan.model_copy(update={"queries": queries})

    def _sanitize_evidence_decision(self, decision: EvidenceDecision) -> EvidenceDecision:
        if decision.sufficient:
            return decision.model_copy(update={"next_queries": []})
        return decision.model_copy(
            update={"next_queries": self._sanitize_query_variants(decision.next_queries)}
        )

    def _sanitize_query_variants(self, queries: list[QueryVariant]) -> list[QueryVariant]:
        sanitized: list[QueryVariant] = []
        seen: set[str] = set()
        for query in queries:
            text = re.sub(r"\s+", " ", query.text).strip()
            if not text:
                continue

            filters_payload = query.filters.model_dump() if query.filters else None
            signature = json.dumps(
                {
                    "text": text.lower(),
                    "mode": query.mode,
                    "filters": filters_payload,
                },
                sort_keys=True,
            )
            if signature in seen:
                continue

            seen.add(signature)
            sanitized.append(query.model_copy(update={"text": text}))
            if len(sanitized) >= self.settings.query_rewrite_limit:
                break
        return sanitized
