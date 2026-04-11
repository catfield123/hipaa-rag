"""Function-calling answering service with explicit retrieval/decision loops."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.string_templates import rag_agent
from app.config import Settings, get_settings
from app.core.exceptions import ConfigurationError
from app.schemas.planning import ResearchDecision
from app.schemas.retrieval import RetrievalEvidence
from app.schemas.types import AgentPipelinePhaseEnum, QueryIntentEnum, RagWsEventType
from app.services.answering_components import agent_utils
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
            phase: AgentPipelinePhaseEnum,
            message: str,
            round_number: int | None = None,
            tool: str | None = None,
        ) -> None:
            """Send one ``type: status`` payload through ``on_status`` when configured.

            Args:
                phase (AgentPipelinePhaseEnum): Pipeline step.
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
                "type": RagWsEventType.STATUS,
                "phase": phase,
                "message": message,
            }
            if round_number is not None:
                payload["round"] = round_number
            if tool is not None:
                payload["tool"] = tool
            await on_status(payload)

        await emit(
            phase=AgentPipelinePhaseEnum.START,
            message=rag_agent.AGENT_STATUS_PREPARING,
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
                phase=AgentPipelinePhaseEnum.PLAN,
                round_number=round_number,
                message=rag_agent.AGENT_STATUS_ROUND_PLAN.format(
                    round_number=round_number,
                    max_rounds=max_rounds,
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
            function_calls, _ = agent_utils.prepare_retrieval_tool_calls(
                raw_function_calls,
                max_queries=max_queries_per_round,
            )
            if not function_calls:
                raise RuntimeError(rag_agent.RUNTIME_RETRIEVAL_ROUND_REQUIRES_TOOLS)

            retrieval_rounds = round_number

            for function_call in function_calls:
                parsed_arguments = agent_utils.parse_tool_call_json_arguments(
                    function_call.function.arguments,
                )
                tool_name = function_call.function.name
                await emit(
                    phase=AgentPipelinePhaseEnum.RETRIEVE,
                    round_number=round_number,
                    tool=tool_name,
                    message=rag_agent.AGENT_STATUS_RETRIEVING.format(
                        round_number=round_number,
                        tool_label=rag_agent.AGENT_TOOL_LABELS.get(
                            tool_name,
                            tool_name,
                        ),
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
                        agent_utils.build_retrieval_history_entry(
                            round_number=round_number,
                            function_name=function_call.function.name,
                            function_args=parsed_arguments,
                            result_count=0,
                            error=str(exc),
                        )
                    )
                    continue

                all_evidence = agent_utils.merge_evidence_by_chunk_id(
                    all_evidence,
                    execution.evidence,
                )
                retrieval_history.append(
                    agent_utils.build_retrieval_history_entry(
                        round_number=round_number,
                        function_name=execution.function_name,
                        function_args=execution.function_args,
                        result_count=len(execution.evidence),
                    )
                )

            await emit(
                phase=AgentPipelinePhaseEnum.DECIDE,
                round_number=round_number,
                message=rag_agent.AGENT_STATUS_DECIDE_CHECK.format(
                    round_number=round_number,
                ),
            )
            latest_decision = await self._decide_next_step(
                question=question,
                evidence=all_evidence,
                round_number=round_number,
            )

            if not latest_decision.continue_retrieval:
                await emit(
                    phase=AgentPipelinePhaseEnum.ANSWER,
                    message=rag_agent.AGENT_STATUS_GENERATING_FINAL,
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

            rationale_snippet = agent_utils.truncate_status_line(latest_decision.rationale)
            await emit(
                phase=AgentPipelinePhaseEnum.DECIDE,
                round_number=round_number,
                message=rag_agent.AGENT_STATUS_MORE_SOURCES.format(
                    round_number=round_number,
                    rationale=rationale_snippet,
                ),
            )

        await emit(
            phase=AgentPipelinePhaseEnum.ANSWER,
            message=rag_agent.AGENT_STATUS_LIMIT_REACHED.format(
                max_rounds=max_rounds,
            ),
        )
        forced_decision = latest_decision or ResearchDecision(
            intent=QueryIntentEnum.GENERAL,
            wants_raw_structure=False,
            continue_retrieval=False,
            rationale=rag_agent.RESEARCH_DECISION_RATIONALE_MAX_ROUNDS,
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
                rag_agent.CONFIG_OPENAI_KEY_REQUIRED_FOR_AGENT,
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
        fallback = rag_agent.FINAL_ANSWER_EMPTY_FALLBACK

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
