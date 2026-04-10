"""Query planning logic for the HIPAA answering pipeline."""

from __future__ import annotations

import json
import logging
import re

from openai import AsyncOpenAI
from pydantic import ValidationError

from app.config import Settings
from app.schemas import QueryPlan, QueryVariant
from app.services.answering_components.common import sanitize_query_plan, sanitize_query_variants
from app.services.answering_components.heuristics import StructuralQueryInterpreter
from app.services.answering_components.prompts import PLANNING_SYSTEM_PROMPT, build_planning_prompt
from app.services.text_utils import tokenize, unique_preserve_order

logger = logging.getLogger(__name__)


class QueryPlanner:
    """Plan retrieval queries with an LLM first and deterministic fallback second."""

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

    async def plan_queries(
        self,
        user_query: str,
        *,
        retrieval_round: int = 1,
        previous_failed_queries: list[str] | None = None,
        intent_hint: str | None = None,
    ) -> QueryPlan:
        """Return a retrieval plan for the user query."""

        fallback = self._fallback_query_plan(user_query)
        if not self.settings.openai_api_key:
            return fallback

        prompt = build_planning_prompt(
            user_query=user_query,
            retrieval_round=retrieval_round,
            previous_failed_queries=previous_failed_queries or [],
            intent_hint=intent_hint,
            query_rewrite_limit=self.settings.query_rewrite_limit,
        )
        messages = [
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.openai_chat_model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.choices[0].message.content or "{}")
            plan = QueryPlan.model_validate(payload)
            return sanitize_query_plan(plan, settings=self.settings) or fallback
        except (json.JSONDecodeError, ValidationError, AttributeError, TypeError) as exc:
            logger.warning("Falling back to heuristic query planning: %s", exc)
            return fallback
        except Exception:
            logger.exception("LLM query planning failed; using heuristic fallback.")
            return fallback

    def _fallback_query_plan(self, user_query: str) -> QueryPlan:
        cleaned = re.sub(r"\bhipaa\b", "", user_query, flags=re.IGNORECASE).strip(" ?")
        keywords = tokenize(cleaned) or tokenize(user_query)
        exact_query = cleaned if cleaned else user_query.strip()
        broad_query = " ".join(unique_preserve_order(keywords[:4])) or exact_query
        inferred_filters = self.interpreter.infer_structural_filters(user_query)
        lowered = user_query.lower()

        if self.interpreter.is_section_text_request(lowered, inferred_filters):
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

        if self.interpreter.is_direct_part_outline_request(lowered, inferred_filters):
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

        if self.interpreter.is_direct_subpart_outline_request(lowered, inferred_filters):
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

        if self.interpreter.is_part_structural_reasoning_request(lowered, inferred_filters):
            return QueryPlan(
                intent="general",
                queries=sanitize_query_variants(
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
                    ],
                    settings=self.settings,
                ),
            )

        if self.interpreter.is_subpart_structural_reasoning_request(lowered, inferred_filters):
            return QueryPlan(
                intent="general",
                queries=sanitize_query_variants(
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
                    ],
                    settings=self.settings,
                ),
            )

        if any(word in lowered for word in ("does", "mention", "is there", "does hipaa")):
            return QueryPlan(
                intent="existence_check",
                queries=sanitize_query_variants(
                    [
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
                    ],
                    settings=self.settings,
                ),
            )

        if any(word in lowered for word in ("quote", "cite", "specific regulation", "full text")):
            return QueryPlan(
                intent="quote_request",
                queries=sanitize_query_variants(
                    [
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
                    ],
                    settings=self.settings,
                ),
            )

        return QueryPlan(
            intent="general",
            queries=sanitize_query_variants(
                [
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
                ],
                settings=self.settings,
            ),
        )
