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
                "allowed_modes": ["bm25_only", "hybrid", "structure_lookup"],
                "bm25_is_required_for_literal_or_mention_queries": True,
                "drop_corpus_obvious_terms_like_hipaa": True,
                "structure_lookup_targets": [
                    "section_text",
                    "part_outline",
                    "subpart_outline",
                ],
                "part_or_subpart_outlines_can_answer_high_level_overview_questions": True,
                "return_raw_structure_only_for_explicit_show_list_full_text_requests": True,
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
                    "Choose structure_lookup when the user explicitly wants the full text of a specific section, "
                    "a list of sections within a subpart, or a part outline with subparts and sections. "
                    "For high-level overview or purpose questions, structure_lookup evidence may still be useful, "
                    "but the final answer should be synthesized rather than returning raw structure text. "
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
        if intent == "structure_lookup" and self._should_return_raw_structure(question) and any(
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
                "choose_structure_lookup_for_direct_section_or_outline_requests": True,
                "structure_lookup_targets": [
                    "section_text",
                    "part_outline",
                    "subpart_outline",
                ],
                "part_or_subpart_outlines_can_answer_high_level_overview_questions": True,
                "if_outline_titles_reveal_the_best_follow_up_section_then_request_section_text": True,
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
                        "mode": "bm25_only | hybrid | structure_lookup",
                        "structure_target": "section_text | part_outline | subpart_outline | null",
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
                    "If no, explain what is missing and propose the next retrieval queries. "
                    "Use structure_lookup when the missing information is a full section text or an outline. "
                    "A part or subpart outline can be sufficient for high-level purpose, scope, coverage, or "
                    "organization questions when the section titles clearly answer the question."
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

        if intent == "structure_lookup" and self._should_return_raw_structure(question):
            return self._render_structural_content(evidence)

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
                "metadata": item.metadata,
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
                "part_or_subpart_outlines_can_support_high_level_answers": True,
                "do_not_dump_large_verbatim_blocks_unless_user_explicitly_requests_full_text": True,
                "prefer_a_concise_direct_answer_followed_by_short_supporting_explanation": True,
            },
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You answer questions about HIPAA using only provided evidence. "
                    "Do not hallucinate. If the question is an existence check, "
                    "answer only from the retrieved BM25 or hybrid evidence. "
                    "Use the structural references in each chunk when citing support. "
                    "Do not paste long evidence blocks verbatim unless the user explicitly asked for the full text."
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
        if self._is_section_text_request(lowered, inferred_filters):
            return QueryPlan(
                intent="structure_lookup",
                queries=[
                    QueryVariant(
                        text=exact_query,
                        mode="structure_lookup",
                        structure_target="section_text",
                        strategy="section_text_lookup",
                        reason="Return the full precomputed text for the cited section.",
                        filters=inferred_filters,
                    )
                ],
            )

        if self._is_direct_part_outline_request(lowered, inferred_filters):
            return QueryPlan(
                intent="structure_lookup",
                queries=[
                    QueryVariant(
                        text=exact_query,
                        mode="structure_lookup",
                        structure_target="part_outline",
                        strategy="part_outline_lookup",
                        reason="Return the part outline with subparts and sections.",
                        filters=inferred_filters,
                    )
                ],
            )

        if self._is_direct_subpart_outline_request(lowered, inferred_filters):
            return QueryPlan(
                intent="structure_lookup",
                queries=[
                    QueryVariant(
                        text=exact_query,
                        mode="structure_lookup",
                        structure_target="subpart_outline",
                        strategy="subpart_outline_lookup",
                        reason="Return the subpart outline with all section headings.",
                        filters=inferred_filters,
                    )
                ],
            )

        if self._is_part_structural_reasoning_request(lowered, inferred_filters):
            return QueryPlan(
                intent="general",
                queries=self._sanitize_query_variants(
                    [
                        QueryVariant(
                            text=exact_query,
                            mode="structure_lookup",
                            structure_target="part_outline",
                            strategy="part_outline_reasoning",
                            reason="Use the part outline as evidence for a high-level answer.",
                            filters=inferred_filters,
                        ),
                        QueryVariant(
                            text=exact_query,
                            mode="hybrid",
                            strategy="part_semantic_follow_up",
                            reason="Retrieve supporting section text within the cited part.",
                            filters=inferred_filters,
                        ),
                    ]
                ),
            )

        if self._is_subpart_structural_reasoning_request(lowered, inferred_filters):
            return QueryPlan(
                intent="general",
                queries=self._sanitize_query_variants(
                    [
                        QueryVariant(
                            text=exact_query,
                            mode="structure_lookup",
                            structure_target="subpart_outline",
                            strategy="subpart_outline_reasoning",
                            reason="Use the subpart outline as evidence for a high-level answer.",
                            filters=inferred_filters,
                        ),
                        QueryVariant(
                            text=exact_query,
                            mode="hybrid",
                            strategy="subpart_semantic_follow_up",
                            reason="Retrieve supporting section text within the cited subpart.",
                            filters=inferred_filters,
                        ),
                    ]
                ),
            )

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
        if intent == "structure_lookup" and self._should_return_raw_structure(question):
            return self._render_structural_content(evidence)

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
                "metadata": item.metadata,
            }
            for item in evidence[:limit]
        ]

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

        if intent == "structure_lookup" and self._should_return_raw_structure(question):
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
            if query.mode == "structure_lookup" and query.structure_target is None:
                continue

            filters_payload = query.filters.model_dump() if query.filters else None
            signature = json.dumps(
                {
                    "text": text.lower(),
                    "mode": query.mode,
                    "structure_target": query.structure_target,
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

    def _render_structural_content(self, evidence: list[RetrievalEvidence]) -> str:
        structural_texts = [
            item.text.strip()
            for item in evidence
            if item.retrieval_mode == "structure_lookup" and item.text.strip()
        ]
        if structural_texts:
            return "\n\n".join(unique_preserve_order(structural_texts))
        return self._fallback_answer("", "general", evidence)

    def _is_section_text_request(
        self,
        lowered_query: str,
        inferred_filters: StructuralFilters | None,
    ) -> bool:
        if not inferred_filters or not inferred_filters.section_number:
            return False
        phrases = (
            "full text",
            "entire section",
            "whole section",
            "text of",
            "show me",
            "show ",
            "give me",
            "provide",
            "quote",
        )
        return any(phrase in lowered_query for phrase in phrases)

    def _is_direct_part_outline_request(
        self,
        lowered_query: str,
        inferred_filters: StructuralFilters | None,
    ) -> bool:
        if not inferred_filters or not inferred_filters.part_number:
            return False
        return self._has_direct_structure_display_signal(lowered_query)

    def _is_direct_subpart_outline_request(
        self,
        lowered_query: str,
        inferred_filters: StructuralFilters | None,
    ) -> bool:
        if not inferred_filters or not inferred_filters.subpart:
            return False
        return self._has_direct_structure_display_signal(lowered_query)

    def _is_part_structural_reasoning_request(
        self,
        lowered_query: str,
        inferred_filters: StructuralFilters | None,
    ) -> bool:
        if not inferred_filters or not inferred_filters.part_number:
            return False
        keywords = (
            "purpose",
            "overview",
            "cover",
            "covers",
            "about",
        )
        return any(keyword in lowered_query for keyword in keywords)

    def _is_subpart_structural_reasoning_request(
        self,
        lowered_query: str,
        inferred_filters: StructuralFilters | None,
    ) -> bool:
        if not inferred_filters or not inferred_filters.subpart:
            return False
        keywords = (
            "purpose",
            "overview",
            "cover",
            "covers",
            "about",
        )
        return any(keyword in lowered_query for keyword in keywords)

    def _should_return_raw_structure(self, question: str) -> bool:
        lowered_query = question.lower()
        inferred_filters = self._infer_structural_filters(question)
        return (
            self._is_section_text_request(lowered_query, inferred_filters)
            or self._is_direct_part_outline_request(lowered_query, inferred_filters)
            or self._is_direct_subpart_outline_request(lowered_query, inferred_filters)
        )

    def _has_direct_structure_display_signal(self, lowered_query: str) -> bool:
        direct_phrases = (
            "list",
            "show",
            "display",
            "outline",
            "contents",
            "table of contents",
            "section headings",
            "all sections",
            "which sections",
        )
        return any(phrase in lowered_query for phrase in direct_phrases)

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
            matching_sections = self._best_outline_sections(question, item)
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
            for section in self._best_outline_sections(question, item)[:2]:
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

        next_queries = self._sanitize_query_variants(next_queries)
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

        ranked_sections = self._best_referenced_sections(question, evidence)
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

        next_queries = self._sanitize_query_variants(next_queries)
        if not next_queries:
            return None

        return EvidenceDecision(
            sufficient=False,
            rationale="Partial chunks point to specific relevant sections, but the full section text is still needed.",
            missing_information=unique_preserve_order(missing_information),
            next_queries=next_queries,
        )

    def _best_outline_sections(
        self,
        question: str,
        evidence: RetrievalEvidence,
    ) -> list[dict[str, object]]:
        candidates = self._outline_sections(evidence)
        if not candidates:
            return []

        question_tokens = set(tokenize(question))
        if not question_tokens:
            question_tokens = {token.lower() for token in re.findall(r"[a-z][a-z0-9']*", question.lower())}

        ranked: list[tuple[int, dict[str, object]]] = []
        for section in candidates:
            section_title = str(section.get("section_title") or "")
            title_tokens = set(tokenize(section_title))
            score = len(question_tokens & title_tokens)
            if "purpose" in question.lower() and "purpose" in section_title.lower():
                score += 3
            if "definition" in question.lower() and "definition" in section_title.lower():
                score += 3
            if "scope" in question.lower() and "scope" in section_title.lower():
                score += 3
            if score > 0:
                ranked.append((score, section))

        ranked.sort(
            key=lambda item: (
                -item[0],
                str(item[1].get("section_number") or ""),
            )
        )
        return [section for _, section in ranked]

    def _best_referenced_sections(
        self,
        question: str,
        evidence: list[RetrievalEvidence],
    ) -> list[dict[str, object]]:
        question_tokens = set(tokenize(question))
        if not question_tokens:
            question_tokens = {token.lower() for token in re.findall(r"[a-z][a-z0-9']*", question.lower())}

        grouped: dict[str, dict[str, object]] = {}
        for item in evidence:
            section_number, section_title = self._parse_section_reference(item.section)
            if not section_number:
                continue

            entry = grouped.setdefault(
                section_number,
                {
                    "section_number": section_number,
                    "section_title": section_title,
                    "score": 0.0,
                    "count": 0,
                },
            )
            entry["count"] = int(entry["count"]) + 1
            entry["score"] = float(entry["score"]) + float(item.score)

            title_tokens = set(tokenize(section_title))
            entry["score"] = float(entry["score"]) + float(len(question_tokens & title_tokens))

            item_text = str(item.text or "").lower()
            if "family" in question.lower() and "family" in item_text:
                entry["score"] = float(entry["score"]) + 2.0
            if "law enforcement" in question.lower() and "law enforcement" in item_text:
                entry["score"] = float(entry["score"]) + 2.0
            if "purpose" in question.lower() and "purpose" in section_title.lower():
                entry["score"] = float(entry["score"]) + 3.0
            if "definition" in question.lower() and "definition" in section_title.lower():
                entry["score"] = float(entry["score"]) + 3.0
            if "scope" in question.lower() and "applicability" in section_title.lower():
                entry["score"] = float(entry["score"]) + 2.0

        ranked = sorted(
            grouped.values(),
            key=lambda item: (
                -float(item["score"]),
                -int(item["count"]),
                str(item["section_number"]),
            ),
        )
        return ranked

    def _parse_section_reference(self, section_label: str | None) -> tuple[str | None, str]:
        if not section_label:
            return None, ""
        match = re.match(r"^§\s*(\d+\.\d+)\b\s*(.*)$", section_label.strip())
        if not match:
            return None, section_label.strip()
        return match.group(1), match.group(2).strip()

    def _outline_sections(self, evidence: RetrievalEvidence) -> list[dict[str, object]]:
        content_type = evidence.metadata.get("content_type")
        if content_type == "subpart_outline":
            sections = evidence.metadata.get("sections")
            return [item for item in sections if isinstance(item, dict)] if isinstance(sections, list) else []

        if content_type == "part_outline":
            sections: list[dict[str, object]] = []
            direct_sections = evidence.metadata.get("direct_sections")
            if isinstance(direct_sections, list):
                sections.extend(item for item in direct_sections if isinstance(item, dict))
            subparts = evidence.metadata.get("subparts")
            if isinstance(subparts, list):
                for subpart in subparts:
                    if not isinstance(subpart, dict):
                        continue
                    subpart_sections = subpart.get("sections")
                    if isinstance(subpart_sections, list):
                        sections.extend(item for item in subpart_sections if isinstance(item, dict))
            return sections

        return []
