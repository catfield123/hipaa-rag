"""Function-calling answering service with explicit retrieval/decision loops."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.schemas.planning import ResearchDecision
from app.schemas.retrieval import RetrievalEvidence
from app.schemas.types import QueryIntentEnum
from app.services.answering_components.decision_functions import (
    DECIDE_RESEARCH_STATUS_FUNCTION_NAME,
    build_research_decision_functions,
    extract_research_decision_payload,
)
from app.services.answering_components.functions import (
    RetrievalFunctionExecutor,
    build_retrieval_functions,
)
from app.services.answering_components.prompts import (
    build_final_answer_messages,
    build_research_decision_messages,
    build_retrieval_round_messages,
)
from app.services.openai_client import get_openai_client
from app.services.retrieval_components import (
    BM25Service,
    DenseRetriever,
    HybridRetriever,
    StructuralContentRetriever,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FunctionAgentResult:
    """Result of a retrieval function-calling run."""

    answer: str
    intent: QueryIntentEnum
    evidence: list[RetrievalEvidence]
    retrieval_rounds: int
    debug_rounds: list[dict[str, object]]


class AnsweringService:
    """Run retrieval rounds until a decision function says the evidence is sufficient."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or get_openai_client()

    async def answer_question(
        self,
        *,
        question: str,
        session: AsyncSession,
        bm25_service: BM25Service,
        dense_retriever: DenseRetriever,
        hybrid_retriever: HybridRetriever,
        structural_retriever: StructuralContentRetriever,
    ) -> FunctionAgentResult:
        """Answer a question through explicit retrieval and decision function loops."""

        self._ensure_openai_configuration()
        function_executor = RetrievalFunctionExecutor(
            session=session,
            bm25_service=bm25_service,
            dense_retriever=dense_retriever,
            hybrid_retriever=hybrid_retriever,
            structural_retriever=structural_retriever,
            default_limit=min(self.settings.retrieval_limit, 8),
        )
        retrieval_functions = build_retrieval_functions(default_limit=min(self.settings.retrieval_limit, 8))

        all_evidence: list[RetrievalEvidence] = []
        retrieval_history: list[dict[str, object]] = []
        debug_rounds: list[dict[str, object]] = []
        retrieval_rounds = 0
        latest_decision: ResearchDecision | None = None
        max_queries_per_round = max(1, self.settings.query_rewrite_limit)

        for round_number in range(1, self.settings.agent_max_rounds + 1):
            retrieval_messages = build_retrieval_round_messages(
                question=question,
                evidence=[item.model_dump() for item in all_evidence],
                retrieval_history=retrieval_history,
                prior_decision=latest_decision.model_dump() if latest_decision is not None else None,
                round_number=round_number,
                broad_query_min=min(3, max_queries_per_round),
                max_queries=max_queries_per_round,
            )
            response = await self.client.chat.completions.create(
                model=self.settings.openai_chat_model,
                messages=retrieval_messages,
                tools=retrieval_functions,
                tool_choice="required",
            )
            message = response.choices[0].message
            raw_function_calls = message.tool_calls or []
            function_calls, skipped_function_calls = _prepare_function_calls(
                raw_function_calls,
                max_queries=max_queries_per_round,
            )
            if not function_calls:
                raise RuntimeError(
                    "Each retrieval round must contain at least one retrieval function call."
                )

            retrieval_rounds = round_number
            round_debug: dict[str, object] = {
                "round": round_number,
                "query_budget": max_queries_per_round,
                "requested_function_call_count": len(raw_function_calls),
                "executed_function_call_count": len(function_calls),
                "function_calls": [],
            }
            if skipped_function_calls:
                round_debug["skipped_function_calls"] = skipped_function_calls

            for function_call in function_calls:
                parsed_arguments = _safe_parse_arguments(function_call.function.arguments)
                try:
                    execution = await function_executor.execute(
                        function_call.function.name,
                        function_call.function.arguments,
                    )
                except Exception as exc:
                    logger.exception("Function execution failed for %s", function_call.function.name)
                    execution_payload = {
                        "function_name": function_call.function.name,
                        "function_args": parsed_arguments,
                        "error": str(exc),
                    }
                    cast_calls = round_debug["function_calls"]
                    assert isinstance(cast_calls, list)
                    cast_calls.append(execution_payload)
                    retrieval_history.append(
                        _build_retrieval_history_entry(
                            round_number=round_number,
                            function_name=function_call.function.name,
                            function_args=parsed_arguments,
                            result_count=0,
                            error=str(exc),
                        )
                    )
                    continue

                all_evidence = _merge_evidence(all_evidence, execution.evidence)
                cast_calls = round_debug["function_calls"]
                assert isinstance(cast_calls, list)
                cast_calls.append(
                    {
                        "function_name": execution.function_name,
                        "function_args": execution.function_args,
                        "result_count": len(execution.evidence),
                    }
                )
                retrieval_history.append(
                    _build_retrieval_history_entry(
                        round_number=round_number,
                        function_name=execution.function_name,
                        function_args=execution.function_args,
                        result_count=len(execution.evidence),
                    )
                )

            latest_decision = await self._decide_next_step(
                question=question,
                evidence=all_evidence,
                round_number=round_number,
            )
            round_debug["total_evidence_count"] = len(all_evidence)
            round_debug["decision"] = latest_decision.model_dump()
            debug_rounds.append(round_debug)

            if not latest_decision.continue_retrieval:
                answer = await self._generate_final_answer(
                    question=question,
                    decision=latest_decision,
                    evidence=all_evidence,
                )
                return FunctionAgentResult(
                    answer=answer,
                    intent=latest_decision.intent,
                    evidence=all_evidence,
                    retrieval_rounds=retrieval_rounds,
                    debug_rounds=debug_rounds,
                )

        forced_decision = latest_decision or ResearchDecision(
            intent=QueryIntentEnum.GENERAL,
            wants_raw_structure=False,
            continue_retrieval=False,
            rationale="Maximum retrieval rounds reached.",
            missing_information=[],
        )
        answer = await self._generate_final_answer(
            question=question,
            decision=forced_decision,
            evidence=all_evidence,
        )
        return FunctionAgentResult(
            answer=answer,
            intent=forced_decision.intent,
            evidence=all_evidence,
            retrieval_rounds=retrieval_rounds,
            debug_rounds=debug_rounds,
        )

    def _ensure_openai_configuration(self) -> None:
        """Reject execution when the required model configuration is missing."""

        if not self.settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required because answering only supports the function-calling agent."
            )

    async def _decide_next_step(
        self,
        *,
        question: str,
        evidence: list[RetrievalEvidence],
        round_number: int,
    ) -> ResearchDecision:
        """Run the explicit post-retrieval decision function."""

        response = await self.client.chat.completions.create(
            model=self.settings.openai_chat_model,
            messages=build_research_decision_messages(
                question=question,
                evidence=[item.model_dump() for item in evidence],
                round_number=round_number,
            ),
            tools=build_research_decision_functions(),
            tool_choice={
                "type": "function",
                "function": {"name": DECIDE_RESEARCH_STATUS_FUNCTION_NAME},
            },
        )
        return ResearchDecision.model_validate(
            extract_research_decision_payload(response.choices[0].message)
        )

    async def _generate_final_answer(
        self,
        *,
        question: str,
        decision: ResearchDecision,
        evidence: list[RetrievalEvidence],
    ) -> str:
        """Generate the final answer after the decision function stops retrieval."""

        response = await self.client.chat.completions.create(
            model=self.settings.openai_chat_model,
            messages=build_final_answer_messages(
                question=question,
                decision=decision.model_dump(),
                evidence=[item.model_dump() for item in evidence],
            ),
        )
        return (response.choices[0].message.content or "").strip() or (
            "I could not produce a final answer from the retrieved evidence."
        )


def _merge_evidence(
    existing: list[RetrievalEvidence],
    new_items: list[RetrievalEvidence],
) -> list[RetrievalEvidence]:
    """Keep evidence unique by chunk id while preserving order."""

    merged: list[RetrievalEvidence] = []
    seen_chunk_ids: set[int] = set()
    for item in [*existing, *new_items]:
        if item.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(item.chunk_id)
        merged.append(item)
    return merged


def _prepare_function_calls(
    function_calls: list[Any],
    *,
    max_queries: int,
) -> tuple[list[Any], list[dict[str, object]]]:
    """Drop exact duplicate calls and cap each round to the configured query budget."""

    prepared_calls: list[Any] = []
    skipped_calls: list[dict[str, object]] = []
    seen_keys: set[str] = set()

    for function_call in function_calls:
        function_name = function_call.function.name
        parsed_arguments = _safe_parse_arguments(function_call.function.arguments)
        call_key = _build_function_call_key(
            function_name=function_name,
            function_args=parsed_arguments,
        )
        if call_key in seen_keys:
            skipped_calls.append(
                {
                    "function_name": function_name,
                    "function_args": parsed_arguments,
                    "reason": "duplicate_exact_call",
                }
            )
            continue
        if len(prepared_calls) >= max_queries:
            skipped_calls.append(
                {
                    "function_name": function_name,
                    "function_args": parsed_arguments,
                    "reason": "over_query_budget",
                }
            )
            continue
        seen_keys.add(call_key)
        prepared_calls.append(function_call)

    return prepared_calls, skipped_calls


def _build_function_call_key(*, function_name: str, function_args: dict[str, Any]) -> str:
    """Return a stable dedupe key for one retrieval function call."""

    normalized_args = _normalize_payload(function_args)
    return json.dumps(
        {"function_name": function_name, "function_args": normalized_args},
        ensure_ascii=True,
        sort_keys=True,
    )


def _build_retrieval_history_entry(
    *,
    round_number: int,
    function_name: str,
    function_args: dict[str, Any],
    result_count: int,
    error: str | None = None,
) -> dict[str, object]:
    """Serialize one retrieval call into compact prompt-friendly history."""

    entry: dict[str, object] = {
        "round": round_number,
        "function_name": function_name,
        "result_count": result_count,
    }
    for key in (
        "query_text",
        "filters",
        "target",
        "section_number",
        "part_number",
        "subpart",
    ):
        value = function_args.get(key)
        if value not in (None, "", [], {}):
            entry[key] = value
    if error is not None:
        entry["error"] = error
    return entry


def _safe_parse_arguments(raw_arguments: str) -> dict[str, Any]:
    """Best-effort JSON parser for tool-call arguments."""

    try:
        parsed = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return {"_raw_arguments": raw_arguments}
    return parsed if isinstance(parsed, dict) else {"_raw_arguments": raw_arguments}


def _normalize_payload(value: Any) -> Any:
    """Normalize payload values so exact duplicate retrieval calls collapse cleanly."""

    if isinstance(value, dict):
        return {
            str(key): _normalize_payload(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        return [_normalize_payload(item) for item in value]
    if isinstance(value, str):
        normalized = " ".join(value.split())
        return normalized.casefold() if normalized else normalized
    return value
