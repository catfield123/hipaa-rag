"""Tool-driven answering loop that lets the LLM choose retrieval operations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.schemas import QueryIntent, RetrievalEvidence
from app.services.retrieval_components import (
    BM25Service,
    DenseRetriever,
    HybridRetriever,
    StructuralContentRetriever,
)
from app.services.answering_components.structure import QuestionStructureAnalyzer
from app.services.answering_components.tools import RetrievalToolExecutor, build_retrieval_tools

logger = logging.getLogger(__name__)

TOOL_AGENT_SYSTEM_PROMPT = (
    "You are a HIPAA question-answering agent with retrieval tools. "
    "Use tools to gather evidence before answering factual questions. "
    "Prefer bm25_search for exact wording, quotes, or mention checks. "
    "Prefer hybrid_search for most semantic questions. "
    "Use dense_search only when broader semantic expansion is useful. "
    "Use get_section_text, list_part_outline, list_subpart_outline, or lookup_structural_content "
    "when the user asks for explicit structural content. "
    "Reuse structural filters when the question cites a part, section, subpart, or marker path. "
    "When you have enough evidence, answer directly and concisely. "
    "If evidence remains insufficient, say so clearly instead of guessing."
)


@dataclass(slots=True)
class ToolAgentResult:
    """Result of a tool-driven answering run."""

    answer: str
    intent: QueryIntent
    evidence: list[RetrievalEvidence]
    retrieval_rounds: int
    debug_rounds: list[dict[str, object]]


class ToolDrivenAnsweringAgent:
    """Run a multi-turn tool-calling loop for retrieval-backed answers."""

    def __init__(
        self,
        *,
        settings: Settings,
        client: AsyncOpenAI,
        structure_analyzer: QuestionStructureAnalyzer,
    ) -> None:
        self.settings = settings
        self.client = client
        self.structure_analyzer = structure_analyzer

    async def answer_question(
        self,
        *,
        question: str,
        session: AsyncSession,
        bm25_service: BM25Service,
        dense_retriever: DenseRetriever,
        hybrid_retriever: HybridRetriever,
        structural_retriever: StructuralContentRetriever,
    ) -> ToolAgentResult:
        """Let the model choose retrieval tools and produce the final answer."""

        structure_analysis = await self.structure_analyzer.analyze_question(question)
        if not self.settings.openai_api_key:
            fallback_answer = (
                "I could not run the retrieval agent because OPENAI_API_KEY is not configured."
            )
            return ToolAgentResult(
                answer=fallback_answer,
                intent=structure_analysis.intent_hint,
                evidence=[],
                retrieval_rounds=0,
                debug_rounds=[],
            )

        tool_executor = RetrievalToolExecutor(
            session=session,
            bm25_service=bm25_service,
            dense_retriever=dense_retriever,
            hybrid_retriever=hybrid_retriever,
            structural_retriever=structural_retriever,
            default_limit=min(self.settings.retrieval_limit, 8),
        )
        tools = build_retrieval_tools(default_limit=min(self.settings.retrieval_limit, 8))
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": TOOL_AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Answer the following HIPAA question.\n\n"
                    f"Question: {question}\n\n"
                    "Question analysis:\n"
                    f"{structure_analysis.model_dump_json()}\n\n"
                    "Use tools when you need retrieval evidence. "
                    "If the user explicitly requested raw structural content, it is acceptable to return it."
                ),
            },
        ]

        all_evidence: list[RetrievalEvidence] = []
        debug_rounds: list[dict[str, object]] = []
        retrieval_rounds = 0

        for round_number in range(1, self.settings.agent_max_rounds + 1):
            response = await self.client.chat.completions.create(
                model=self.settings.openai_chat_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            message = response.choices[0].message
            tool_calls = message.tool_calls or []
            assistant_payload: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            if tool_calls:
                assistant_payload["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in tool_calls
                ]
            messages.append(assistant_payload)

            if not tool_calls:
                return ToolAgentResult(
                    answer=(message.content or "").strip()
                    or "I could not produce a final answer from the retrieved evidence.",
                    intent=structure_analysis.intent_hint,
                    evidence=all_evidence,
                    retrieval_rounds=retrieval_rounds,
                    debug_rounds=debug_rounds,
                )

            retrieval_rounds = round_number
            round_debug: dict[str, object] = {
                "round": round_number,
                "tool_calls": [],
                "assistant_content": message.content or "",
            }

            for tool_call in tool_calls:
                try:
                    execution = await tool_executor.execute(
                        tool_call.function.name,
                        tool_call.function.arguments,
                    )
                except Exception as exc:
                    logger.exception("Tool execution failed for %s", tool_call.function.name)
                    execution_payload = {
                        "tool_name": tool_call.function.name,
                        "error": str(exc),
                    }
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(execution_payload),
                        }
                    )
                    cast_calls = round_debug["tool_calls"]
                    assert isinstance(cast_calls, list)
                    cast_calls.append(execution_payload)
                    continue

                all_evidence = _merge_evidence(all_evidence, execution.evidence)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": execution.content,
                    }
                )
                cast_calls = round_debug["tool_calls"]
                assert isinstance(cast_calls, list)
                cast_calls.append(
                    {
                        "tool_name": execution.tool_name,
                        "tool_args": execution.tool_args,
                        "result_count": len(execution.evidence),
                    }
                )

            round_debug["total_evidence_count"] = len(all_evidence)
            debug_rounds.append(round_debug)

        follow_up_messages = [
            *messages,
            {
                "role": "system",
                "content": (
                    "Stop calling tools. Provide the best final answer you can from the evidence already retrieved. "
                    "If the evidence is still insufficient, say that clearly."
                ),
            },
        ]
        response = await self.client.chat.completions.create(
            model=self.settings.openai_chat_model,
            messages=follow_up_messages,
        )
        return ToolAgentResult(
            answer=(response.choices[0].message.content or "").strip()
            or "I could not produce a final answer from the retrieved evidence.",
            intent=structure_analysis.intent_hint,
            evidence=all_evidence,
            retrieval_rounds=retrieval_rounds,
            debug_rounds=debug_rounds,
        )


def _merge_evidence(
    existing: list[RetrievalEvidence],
    new_items: list[RetrievalEvidence],
) -> list[RetrievalEvidence]:
    merged: list[RetrievalEvidence] = []
    seen_chunk_ids: set[int] = set()
    for item in [*existing, *new_items]:
        if item.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(item.chunk_id)
        merged.append(item)
    return merged
