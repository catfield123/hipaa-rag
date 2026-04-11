"""Function-calling answering service with explicit retrieval/decision loops."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.core.exceptions import ConfigurationError
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

AgentStatusEmitter = Callable[[dict[str, Any]], Awaitable[None]]
AgentAnswerDeltaEmitter = Callable[[str], Awaitable[None]]

_TOOL_LABELS: dict[str, str] = {
    "hybrid_search": "hybrid search",
    "bm25_search": "keyword match (BM25)",
    "lookup_structural_content": "structural content",
    "get_section_text": "section text",
    "list_part_outline": "part outline",
    "list_subpart_outline": "subpart outline",
}


def _tool_label(function_name: str) -> str:
    """Map internal tool names to short user-facing labels for status messages.

    Args:
        function_name (str): OpenAI tool name (e.g. ``hybrid_search``).

    Returns:
        str: Human-readable label, or ``function_name`` if unknown.

    Raises:
        None
    """

    return _TOOL_LABELS.get(function_name, function_name)


def _truncate_text(text: str, *, limit: int = 280) -> str:
    """Collapse whitespace and truncate a string for compact status lines.

    Args:
        text (str): Arbitrary text (e.g. model rationale).
        limit (int): Maximum grapheme length before appending an ellipsis.

    Returns:
        str: Single-line truncated string.

    Raises:
        None
    """

    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "…"


@dataclass(slots=True)
class FunctionAgentResult:
    """Result of a retrieval function-calling run.

    Args (fields):
        answer (str): Final model answer string.
        intent (QueryIntentEnum): Intent from the research decision.
        evidence (list[RetrievalEvidence]): Merged deduplicated evidence across rounds.
        retrieval_rounds (int): Number of retrieval rounds that executed at least one tool call.
    """

    answer: str
    intent: QueryIntentEnum
    evidence: list[RetrievalEvidence]
    retrieval_rounds: int


class AnsweringService:
    """Run retrieval rounds until a decision function says the evidence is sufficient."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        """Create an answering service with optional overrides for tests.

        Args:
            settings (Settings | None): Application settings; defaults to :func:`get_settings`.
            client (AsyncOpenAI | None): OpenAI async client; defaults to :func:`get_openai_client`.

        Returns:
            None

        Raises:
            None
        """

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
        on_status: AgentStatusEmitter | None = None,
        on_answer_delta: AgentAnswerDeltaEmitter | None = None,
    ) -> FunctionAgentResult:
        """Answer a question through explicit retrieval and decision function loops.

        Args:
            question (str): User question text.
            session (AsyncSession): Async DB session for retrievers.
            bm25_service (BM25Service): Lexical search backend.
            dense_retriever (DenseRetriever): Dense vector backend.
            hybrid_retriever (HybridRetriever): Hybrid fusion backend.
            structural_retriever (StructuralContentRetriever): Structural lookup backend.
            on_status (AgentStatusEmitter | None): Optional async callback for UI status dicts.
            on_answer_delta (AgentAnswerDeltaEmitter | None): Optional streaming answer chunks.

        Returns:
            FunctionAgentResult: Final answer text, intent, merged evidence, and round count.

        Raises:
            ConfigurationError: If ``OPENAI_API_KEY`` is not configured.
            RuntimeError: If a retrieval round produces no tool calls when calls are required.
        """

        self._ensure_openai_configuration()
        max_rounds = self.settings.agent_max_rounds

        async def emit(
            *,
            phase: str,
            message: str,
            round_number: int | None = None,
            tool: str | None = None,
        ) -> None:
            """Send one ``type: status`` payload through ``on_status`` when configured.

            Args:
                phase (str): Pipeline phase (e.g. ``start``, ``plan``, ``retrieve``, ``decide``, ``answer``).
                message (str): User-visible status line.
                round_number (int | None): Retrieval round index, or ``None`` when not applicable.
                tool (str | None): Active tool name during ``retrieve``, or ``None``.

            Returns:
                None

            Raises:
                None
            """

            if on_status is None:
                return
            payload: dict[str, Any] = {
                "type": "status",
                "phase": phase,
                "message": message,
            }
            if round_number is not None:
                payload["round"] = round_number
            if tool is not None:
                payload["tool"] = tool
            await on_status(payload)

        await emit(
            phase="start",
            message="Preparing the agent and HIPAA knowledge-base search…",
        )
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
        retrieval_rounds = 0
        latest_decision: ResearchDecision | None = None
        max_queries_per_round = max(1, self.settings.query_rewrite_limit)

        for round_number in range(1, max_rounds + 1):
            await emit(
                phase="plan",
                round_number=round_number,
                message=(
                    f"Round {round_number} of {max_rounds}: choosing retrieval tool queries…"
                ),
            )
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
            function_calls, _ = _prepare_function_calls(
                raw_function_calls,
                max_queries=max_queries_per_round,
            )
            if not function_calls:
                raise RuntimeError(
                    "Each retrieval round must contain at least one retrieval function call."
                )

            retrieval_rounds = round_number

            for function_call in function_calls:
                parsed_arguments = _safe_parse_arguments(function_call.function.arguments)
                tool_name = function_call.function.name
                await emit(
                    phase="retrieve",
                    round_number=round_number,
                    tool=tool_name,
                    message=(
                        f"Round {round_number}: retrieving — {_tool_label(tool_name)}…"
                    ),
                )
                try:
                    execution = await function_executor.execute(
                        function_call.function.name,
                        function_call.function.arguments,
                    )
                except Exception as exc:
                    logger.exception("Function execution failed for %s", function_call.function.name)
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
                retrieval_history.append(
                    _build_retrieval_history_entry(
                        round_number=round_number,
                        function_name=execution.function_name,
                        function_args=execution.function_args,
                        result_count=len(execution.evidence),
                    )
                )

            await emit(
                phase="decide",
                round_number=round_number,
                message=f"Round {round_number}: checking whether evidence is sufficient…",
            )
            latest_decision = await self._decide_next_step(
                question=question,
                evidence=all_evidence,
                round_number=round_number,
            )

            if not latest_decision.continue_retrieval:
                await emit(
                    phase="answer",
                    message="Generating the final answer from retrieved sources…",
                )
                answer = await self._generate_final_answer(
                    question=question,
                    decision=latest_decision,
                    evidence=all_evidence,
                    on_delta=on_answer_delta,
                )
                return FunctionAgentResult(
                    answer=answer,
                    intent=latest_decision.intent,
                    evidence=all_evidence,
                    retrieval_rounds=retrieval_rounds,
                )

            rationale_snippet = _truncate_text(latest_decision.rationale)
            await emit(
                phase="decide",
                round_number=round_number,
                message=(
                    f"Round {round_number}: more sources needed. "
                    f"Rationale: {rationale_snippet}"
                ),
            )

        await emit(
            phase="answer",
            message=(
                f"Retrieval round limit reached ({max_rounds}). Generating an answer from available sources…"
            ),
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
            on_delta=on_answer_delta,
        )
        return FunctionAgentResult(
            answer=answer,
            intent=forced_decision.intent,
            evidence=all_evidence,
            retrieval_rounds=retrieval_rounds,
        )

    def _ensure_openai_configuration(self) -> None:
        """Reject execution when the required model configuration is missing.

        Args:
            None

        Returns:
            None

        Raises:
            ConfigurationError: If the OpenAI API key is empty.
        """

        if not self.settings.openai_api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY is required because answering only supports the function-calling agent."
            )

    async def _decide_next_step(
        self,
        *,
        question: str,
        evidence: list[RetrievalEvidence],
        round_number: int,
    ) -> ResearchDecision:
        """Call ``decide_research_status`` to determine whether to continue retrieval.

        Args:
            question (str): User question text.
            evidence (list[RetrievalEvidence]): Evidence gathered through the current round.
            round_number (int): Completed retrieval round index.

        Returns:
            ResearchDecision: Validated decision payload from the model tool call.

        Raises:
            ValueError: If the expected tool call is missing (see :func:`extract_research_decision_payload`).
            ValidationError: If the tool arguments do not match :class:`~app.schemas.planning.ResearchDecision`.
        """

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
        on_delta: AgentAnswerDeltaEmitter | None = None,
    ) -> str:
        """Generate the user-visible answer, optionally streaming token deltas.

        Args:
            question (str): User question text.
            decision (ResearchDecision): Final research decision (intent, structure flags).
            evidence (list[RetrievalEvidence]): Grounding excerpts passed into the prompt.
            on_delta (AgentAnswerDeltaEmitter | None): If set, streams answer fragments from the chat completion.

        Returns:
            str: Final answer text (non-empty, or a configured fallback when the model returns nothing).

        Raises:
            ConfigurationError: If OpenAI is misconfigured (propagated from the client).
            Exception: Network or API errors from ``chat.completions.create``.
        """

        messages = build_final_answer_messages(
            question=question,
            decision=decision.model_dump(),
            evidence=[item.model_dump() for item in evidence],
        )
        fallback = "I could not produce a final answer from the retrieved evidence."

        if on_delta is None:
            response = await self.client.chat.completions.create(
                model=self.settings.openai_chat_model,
                messages=messages,
            )
            return (response.choices[0].message.content or "").strip() or fallback

        stream = await self.client.chat.completions.create(
            model=self.settings.openai_chat_model,
            messages=messages,
            stream=True,
        )
        parts: list[str] = []
        async for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if choice is None:
                continue
            piece = choice.delta.content
            if piece:
                parts.append(piece)
                await on_delta(piece)

        text = "".join(parts).strip() or fallback
        return text


def _merge_evidence(
    existing: list[RetrievalEvidence],
    new_items: list[RetrievalEvidence],
) -> list[RetrievalEvidence]:
    """Merge evidence lists, deduplicating by ``chunk_id`` while preserving first-seen order.

    Args:
        existing (list[RetrievalEvidence]): Prior rounds' evidence.
        new_items (list[RetrievalEvidence]): New hits from the current round.

    Returns:
        list[RetrievalEvidence]: Combined list without duplicate chunk ids.

    Raises:
        None
    """

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
    """Deduplicate tool calls by normalized arguments and enforce a per-round call budget.

    Args:
        function_calls (list[Any]): Raw ``tool_calls`` entries from the chat completion.
        max_queries (int): Maximum distinct calls to keep this round.

    Returns:
        tuple[list[Any], list[dict[str, object]]]: Kept call objects and skipped-call diagnostics.

    Raises:
        None
    """

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
    """Return a stable JSON key for deduplicating identical retrieval calls.

    Args:
        function_name (str): Tool name.
        function_args (dict[str, Any]): Parsed arguments (after :func:`_safe_parse_arguments`).

    Returns:
        str: Canonical JSON string for set membership checks.

    Raises:
        None
    """

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
    """Serialize one retrieval tool invocation for inclusion in the next round's user prompt.

    Args:
        round_number (int): Round index when the call was made.
        function_name (str): Tool name executed.
        function_args (dict[str, Any]): Parsed arguments.
        result_count (int): Number of evidence rows returned (``0`` on failure).
        error (str | None): Error message when execution failed.

    Returns:
        dict[str, object]: Compact history dict (query fields omitted when empty).

    Raises:
        None
    """

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
    """Parse tool-call JSON; on failure return a dict capturing the raw string.

    Args:
        raw_arguments (str): JSON object string from the OpenAI tool call.

    Returns:
        dict[str, Any]: Parsed object, or ``{\"_raw_arguments\": ...}`` when JSON is invalid.

    Raises:
        None
    """

    try:
        parsed = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return {"_raw_arguments": raw_arguments}
    return parsed if isinstance(parsed, dict) else {"_raw_arguments": raw_arguments}


def _normalize_payload(value: Any) -> Any:
    """Recursively normalize dict/list/string values for stable dedupe keys.

    Args:
        value (Any): Argument subtree from a tool call.

    Returns:
        Any: Structure with sorted dict keys, normalized strings (casefold, collapsed whitespace).

    Raises:
        None
    """

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
