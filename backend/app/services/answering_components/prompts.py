"""Prompt templates for the answering components package."""

from __future__ import annotations

import json
from typing import Any

FUNCTION_AGENT_SYSTEM_PROMPT = (
    "You are in the retrieval phase of a HIPAA answering agent. "
    "Do not answer the question in plain text during this phase. "
    "Call one or more retrieval functions to gather the next useful evidence. "
    "Your retrieval plan must stay strictly grounded in the database-backed evidence you can fetch through functions. "
    "Prefer bm25_search for exact wording, quotes, or mention checks. "
    "Prefer hybrid_search for most semantic questions. "
    "Use dense_search only when broader semantic expansion is useful. "
    "Use get_section_text, list_part_outline, list_subpart_outline, or lookup_structural_content "
    "when the user asks for explicit structural content. "
    "Reuse structural filters when the question cites a part, section, subpart, or marker path. "
    "If the question asks which sections, rules, or provisions apply, retrieve those actual sections or rules. "
    "Definitions or tangential context alone are not enough for that kind of answer. "
    "Prefer queries that close the most important evidence gaps first."
)

RESEARCH_DECISION_SYSTEM_PROMPT = (
    "You are in the post-retrieval decision phase of a HIPAA answering agent. "
    "You must decide whether the current evidence is sufficient or whether another retrieval round is needed. "
    "Respond only by calling the decide_research_status function. "
    "Set continue_retrieval=true when the current evidence still misses facts needed for a grounded answer. "
    "Set continue_retrieval=false only when the evidence is sufficient for the final answer, "
    "Never stop if answering would require naming sections, provisions, or obligations that do not appear in the evidence. "
    "If the user asks which sections apply and the evidence only contains definitions, examples, or indirect context, "
    "set continue_retrieval=true. "
    "If the exact applicable sections are not explicitly present in the evidence, do not mark the evidence sufficient."
)

FINAL_ANSWER_SYSTEM_PROMPT = (
    "You are in the final answer phase of a HIPAA answering agent. "
    "Answer only from the provided evidence. "
    "Stay strictly grounded in the retrieved database evidence. "
    "Do not use your own HIPAA knowledge to fill gaps. "
    "Do not mention any section, subsection, part, subpart, requirement, or factual claim unless it appears in the evidence. "
    "Do not infer missing citations. Do not add likely sections from memory. "
    "If the evidence is insufficient, say so clearly instead of guessing. "
    "If the user asks which sections apply and the evidence does not explicitly contain those sections, "
    "say that the retrieved evidence is insufficient to identify the exact applicable sections. "
    "If the decision says wants_raw_structure=true, return the requested structural content directly and cleanly. "
    "Otherwise provide a concise direct answer followed by short supporting explanation."
)


def build_retrieval_round_messages(
    *,
    question: str,
    evidence: list[dict[str, Any]],
    prior_decision: dict[str, Any] | None,
    round_number: int,
) -> list[dict[str, str]]:
    """Build messages for one retrieval round."""

    return [
        {"role": "system", "content": FUNCTION_AGENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Retrieval round: {round_number}\n\n"
                f"Question:\n{question}\n\n"
                "Evidence already collected:\n"
                f"{json.dumps(evidence, ensure_ascii=True)}\n\n"
                "Previous decision:\n"
                f"{json.dumps(prior_decision, ensure_ascii=True)}"
            ),
        },
    ]


def build_research_decision_messages(
    *,
    question: str,
    evidence: list[dict[str, Any]],
    round_number: int,
) -> list[dict[str, str]]:
    """Build messages for the decision step after one retrieval round."""

    return [
        {"role": "system", "content": RESEARCH_DECISION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Decision round after retrieval round: {round_number}\n\n"
                f"Question:\n{question}\n\n"
                "Supported sections explicitly present in evidence metadata or text:\n"
                f"{json.dumps(_build_supported_sections(evidence), ensure_ascii=True)}\n\n"
                "Current evidence:\n"
                f"{json.dumps(evidence, ensure_ascii=True)}"
            ),
        },
    ]


def build_final_answer_messages(
    *,
    question: str,
    evidence: list[dict[str, Any]],
    decision: dict[str, Any],
) -> list[dict[str, str]]:
    """Build messages for final answer generation."""

    return [
        {"role": "system", "content": FINAL_ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                "Decision:\n"
                f"{json.dumps(decision, ensure_ascii=True)}\n\n"
                "Supported sections explicitly present in the evidence:\n"
                f"{json.dumps(_build_supported_sections(evidence), ensure_ascii=True)}\n\n"
                "Evidence:\n"
                f"{json.dumps(evidence, ensure_ascii=True)}"
            ),
        },
    ]


def _build_supported_sections(evidence: list[dict[str, Any]]) -> list[str]:
    """Extract unique section labels already present in the evidence payload."""

    sections = []
    seen: set[str] = set()
    for item in evidence:
        for candidate in (
            item.get("section"),
            item.get("path_text"),
        ):
            if not candidate:
                continue
            normalized = str(candidate).strip()
            if normalized in seen:
                continue
            seen.add(normalized)
            sections.append(normalized)
    return sections
