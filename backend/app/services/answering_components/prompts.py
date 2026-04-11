"""Prompt templates for the answering components package."""

from __future__ import annotations

import json
from typing import Any

from app.schemas.types import QueryIntentEnum
from app.string_templates import llm_user

FUNCTION_AGENT_SYSTEM_PROMPT = (
    "You are in the retrieval phase of a HIPAA answering agent. "
    "Do not answer the question in plain text during this phase. "
    "Call one or more retrieval functions to gather the next useful evidence. "
    "Your retrieval plan must stay strictly grounded in the database-backed evidence you can fetch through functions. "
    "Use hybrid_search as the default search tool for almost all text retrieval. "
    "For ordinary legal or policy questions about permissions, requirements, disclosures, obligations, or exceptions, "
    "start with hybrid_search even if the question contains terms that could also be matched lexically. "
    "Use bm25_search only when the task is explicitly to verify exact wording, quotes, literal mentions, or precise mention checks. "
    "Use get_section_text, list_part_outline, list_subpart_outline, or lookup_structural_content "
    "when the user asks for explicit structural content. "
    "Reuse structural filters when the question cites a part, section, subpart, or marker path. "
    "If the question asks which sections, rules, or provisions apply, retrieve those actual sections or rules. "
    "Definitions or tangential context alone are not enough for that kind of answer. "
    "Prefer queries that close the most important evidence gaps first. "
    "When the evidence is still broad or incomplete, issue multiple distinct hybrid_search calls that attack the problem "
    "from different angles instead of producing one or two light paraphrases. "
    "Treat prior retrieval calls as history: do not repeat the same query_text or a near-duplicate wording unless you are "
    "adding materially different filters or testing a genuinely new retrieval hypothesis. "
    "If the evidence already identifies one concrete section, part, or subpart and only a narrow follow-up is needed, "
    "one or two focused structural calls are acceptable; otherwise use a diversified batch of searches."
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
    "Answer using only the regulatory excerpts supplied in the user message (the JSON list labeled with excerpts). "
    "Stay strictly grounded in that material. "
    "Do not use your own HIPAA knowledge to fill gaps. "
    "Do not mention any section, subsection, part, subpart, requirement, or factual claim unless it appears in those excerpts. "
    "Do not infer missing citations. Do not add likely sections from memory. "
    "If the supplied material is insufficient to answer, say so clearly instead of guessing. "
    "If the user asks which sections apply and the supplied excerpts do not explicitly contain those sections, "
    "say that the provided text does not identify the exact applicable sections. "
    "In the reply visible to the user, use plain professional language. Do not use internal or technical terms such as "
    "'evidence', 'retrieval', 'chunks', 'the database', 'RAG', or 'payload'. Prefer wording such as "
    "'the provided regulatory text', 'these excerpts', 'the cited sections', 'based on the material provided', or "
    "'the available text does not include …'. "
    "If the decision says wants_raw_structure=true, return the requested structural content directly and cleanly. "
    "LENGTH AND DEPTH: Calibrate length and detail to the question. "
    "For high-level asks—overall purpose, main goal, summary, scope, 'in brief', 'at a high level', what a part or subpart "
    "is mainly about—give a short direct answer (typically a few sentences or one short paragraph). "
    "Paraphrase the idea; name the relevant sections or subparts by citation; do not paste multi-paragraph statutory text "
    "or enumerate every sub-rule unless the user clearly asks for comprehensive detail, full wording, or a full list. "
    "For questions that explicitly request verbatim text, direct quotes, every condition, all exceptions, exhaustive lists, "
    "or step-by-step detail, a longer structured answer is appropriate. "
    "Infer desired depth and length only from the user's wording and intent—there is no separate length setting. "
    "When you do include exact regulatory wording (because the question calls for quotes, precise language, or short illustration), "
    "reproduce the relevant `text` field exactly as provided. Do not replace missing parts with bracketed placeholders such as "
    "[conditions apply], [purposes], or similar invented fillers. Do not paraphrase inside passages you present as direct quotations. "
    "If you summarize, label it clearly as a summary separate from quoted text. "
    "If a passage is clearly truncated or incomplete in the supplied material, say that the excerpt is partial "
    "and quote only what was provided."
)

FINAL_ANSWER_QUOTE_REQUEST_SUPPLEMENT = (
    "The user asked for verbatim regulatory text (citation / quote request). "
    "For each relevant row in the supplied JSON, show the `path_text` (or section plus markers) then the full `text` field "
    "exactly as stored—no inline rewriting, no bracketed omissions, no '[conditions apply]' style shortcuts. "
    "If multiple excerpts apply, list them as separate blocks in the order given below. "
    "If an excerpt does not contain the full rule, say so in user-facing language after quoting what was provided "
    "(without calling it 'evidence' or 'the database')."
)


def build_retrieval_round_messages(
    *,
    question: str,
    evidence: list[dict[str, Any]],
    retrieval_history: list[dict[str, object]],
    prior_decision: dict[str, Any] | None,
    round_number: int,
    broad_query_min: int,
    max_queries: int,
) -> list[dict[str, str]]:
    """Assemble chat messages for one retrieval round (system + user with context JSON).

    Args:
        question (str): End-user question text.
        evidence (list[dict[str, Any]]): Serialized evidence accumulated so far.
        retrieval_history (list[dict[str, object]]): Prior tool calls and outcomes.
        prior_decision (dict[str, Any] | None): Last research decision, if any.
        round_number (int): 1-based round index.
        broad_query_min (int): Minimum broad queries suggested in the user prompt.
        max_queries (int): Maximum tool calls allowed this round.

    Returns:
        list[dict[str, str]]: OpenAI ``messages`` payload (roles ``system`` / ``user``).

    Raises:
        None
    """

    return [
        {"role": "system", "content": FUNCTION_AGENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": llm_user.RETRIEVAL_ROUND.format(
                round_number=round_number,
                question=question,
                broad_query_min=broad_query_min,
                max_queries=max_queries,
                evidence_json=json.dumps(evidence, ensure_ascii=True),
                retrieval_history_json=json.dumps(retrieval_history, ensure_ascii=True),
                prior_decision_json=json.dumps(prior_decision, ensure_ascii=True),
            ),
        },
    ]


def build_research_decision_messages(
    *,
    question: str,
    evidence: list[dict[str, Any]],
    round_number: int,
) -> list[dict[str, str]]:
    """Assemble chat messages for the ``decide_research_status`` tool call after a retrieval round.

    Args:
        question (str): End-user question text.
        evidence (list[dict[str, Any]]): Evidence from rounds up to and including this step.
        round_number (int): Round index just completed.

    Returns:
        list[dict[str, str]]: OpenAI ``messages`` for the decision-only completion.

    Raises:
        None
    """

    return [
        {"role": "system", "content": RESEARCH_DECISION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": llm_user.RESEARCH_DECISION.format(
                round_number=round_number,
                question=question,
                supported_sections_json=json.dumps(
                    _build_supported_sections(evidence),
                    ensure_ascii=True,
                ),
                evidence_json=json.dumps(evidence, ensure_ascii=True),
            ),
        },
    ]


def _final_answer_system_content(decision: dict[str, Any]) -> str:
    """Choose the final-answer system prompt, appending quote rules for ``quote_request`` intent.

    Args:
        decision (dict[str, Any]): Serialized :class:`~app.schemas.planning.ResearchDecision` (includes ``intent``).

    Returns:
        str: Full system prompt string for the final completion.

    Raises:
        None
    """

    content = FINAL_ANSWER_SYSTEM_PROMPT
    intent = decision.get("intent")
    if intent == QueryIntentEnum.QUOTE_REQUEST:
        content = llm_user.FINAL_ANSWER_SYSTEM_WITH_QUOTE_SUPPLEMENT.format(
            system_prompt=content,
            quote_supplement=FINAL_ANSWER_QUOTE_REQUEST_SUPPLEMENT,
        )
    return content


def build_final_answer_messages(
    *,
    question: str,
    evidence: list[dict[str, Any]],
    decision: dict[str, Any],
) -> list[dict[str, str]]:
    """Assemble chat messages for the final natural-language answer (grounded in JSON excerpts).

    Args:
        question (str): End-user question text.
        evidence (list[dict[str, Any]]): Serialized retrieval evidence rows.
        decision (dict[str, Any]): Research decision payload (intent, structure flags, etc.).

    Returns:
        list[dict[str, str]]: OpenAI ``messages`` for non-streaming or streaming final completion.

    Raises:
        None
    """

    return [
        {"role": "system", "content": _final_answer_system_content(decision)},
        {
            "role": "user",
            "content": llm_user.FINAL_ANSWER.format(
                question=question,
                decision_json=json.dumps(decision, ensure_ascii=True),
                supported_sections_json=json.dumps(
                    _build_supported_sections(evidence),
                    ensure_ascii=True,
                ),
                evidence_json=json.dumps(evidence, ensure_ascii=True),
            ),
        },
    ]


def _build_supported_sections(evidence: list[dict[str, Any]]) -> list[str]:
    """Collect unique section labels from evidence ``section`` and ``path_text`` fields.

    Args:
        evidence (list[dict[str, Any]]): Evidence dicts (typically ``model_dump()`` shapes).

    Returns:
        list[str]: De-duplicated labels in first-seen order.

    Raises:
        None
    """

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
