"""Human-readable prompt builders for the answering pipeline."""

from __future__ import annotations

import json

from app.schemas import QueryVariant, RetrievalEvidence

PLANNING_SYSTEM_PROMPT = (
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
)

EVIDENCE_SYSTEM_PROMPT = (
    "You are a legal retrieval judge for HIPAA question answering. "
    "Return JSON only. Decide whether the currently retrieved evidence is enough to answer "
    "the user's question. If yes, set sufficient=true and next_queries=[]. "
    "If no, explain what is missing and propose the next retrieval queries. "
    "Use structure_lookup when the missing information is a full section text or an outline. "
    "A part or subpart outline can be sufficient for high-level purpose, scope, coverage, or "
    "organization questions when the section titles clearly answer the question."
)

ANSWER_SYSTEM_PROMPT = (
    "You answer questions about HIPAA using only provided evidence. "
    "Do not hallucinate. If the question is an existence check, "
    "answer only from the retrieved BM25 or hybrid evidence. "
    "Use the structural references in each chunk when citing support. "
    "Do not paste long evidence blocks verbatim unless the user explicitly asked for the full text."
)


def build_planning_prompt(
    *,
    user_query: str,
    retrieval_round: int,
    previous_failed_queries: list[str],
    intent_hint: str | None,
    query_rewrite_limit: int,
) -> str:
    """Build the planner user prompt as plain text plus compact JSON context."""

    context = {
        "user_query": user_query,
        "retrieval_round": retrieval_round,
        "previous_failed_queries": previous_failed_queries,
        "intent_hint": intent_hint,
    }
    return (
        "Plan retrieval queries for the request below.\n\n"
        "Requirements:\n"
        f"- Return between 1 and {query_rewrite_limit} queries.\n"
        "- Allowed modes: bm25_only, hybrid, structure_lookup.\n"
        "- Use bm25_only for literal, quote, mention, or exact existence checks.\n"
        "- Use hybrid for semantic, conceptual, or multi-section questions.\n"
        "- Use structure_lookup only when the user explicitly wants full section text, "
        "a subpart outline, or a part outline.\n"
        "- Part and subpart outlines may support high-level overview questions, but those should still "
        "usually lead to a synthesized answer rather than dumping raw structure.\n"
        "- Return raw structure only for explicit show/list/full-text requests.\n"
        "- Drop corpus-obvious filler terms like HIPAA when they do not improve retrieval.\n"
        "- If the user mentions a specific part, section, subpart, or marker path, add filters.\n"
        "- Optional filters may include: part_number, section_number, subpart, marker_path.\n\n"
        "Return JSON only using the expected QueryPlan shape.\n\n"
        "Request context:\n"
        f"{json.dumps(context, ensure_ascii=True)}"
    )


def build_evidence_prompt(
    *,
    question: str,
    intent: str,
    retrieval_round: int,
    max_rounds: int,
    attempted_queries: list[QueryVariant],
    evidence_payload: list[dict[str, object]],
    query_rewrite_limit: int,
) -> str:
    """Build the evidence-assessment prompt as readable text plus JSON context."""

    context = {
        "question": question,
        "intent": intent,
        "retrieval_round": retrieval_round,
        "max_rounds": max_rounds,
        "attempted_queries": [query.model_dump() for query in attempted_queries],
        "evidence": evidence_payload,
    }
    return (
        "Decide whether the currently retrieved evidence is enough to answer the question.\n\n"
        "Requirements:\n"
        "- The answer must be fully supported by the retrieved evidence.\n"
        "- If the evidence is sufficient, return sufficient=true and next_queries=[].\n"
        f"- If the evidence is insufficient, return between 1 and {query_rewrite_limit} follow-up queries.\n"
        "- Avoid repeating or lightly rephrasing queries that were already attempted.\n"
        "- Use bm25_only for literal, quote, or exact mention checks.\n"
        "- Use hybrid for semantic follow-up retrieval.\n"
        "- Use structure_lookup for direct section text or outline retrieval.\n"
        "- Valid structure_lookup targets are: section_text, part_outline, subpart_outline.\n"
        "- Part and subpart outlines can be sufficient for high-level overview questions.\n"
        "- If an outline title clearly points to the needed section, request that section's full text.\n"
        "- Optional filters may include: part_number, section_number, subpart, marker_path.\n\n"
        "Return JSON only with these fields: sufficient, rationale, missing_information, next_queries.\n"
        "Each next query must include: text, mode, structure_target, strategy, reason, filters.\n\n"
        "Assessment context:\n"
        f"{json.dumps(context, ensure_ascii=True)}"
    )


def build_answer_prompt(
    *,
    question: str,
    intent: str,
    evidence: list[RetrievalEvidence],
) -> str:
    """Build the answer-synthesis prompt as readable text plus JSON evidence."""

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
    context = {
        "question": question,
        "intent": intent,
        "evidence": evidence_payload,
    }
    return (
        "Answer the user's HIPAA question using only the provided evidence.\n\n"
        "Requirements:\n"
        "- Use only the provided evidence and do not guess.\n"
        "- If the evidence is insufficient, say that clearly.\n"
        "- Be explicit when answering negative or existence-check questions.\n"
        "- Part and subpart outlines may support high-level overview answers.\n"
        "- Do not dump large verbatim passages unless the user explicitly asked for full text.\n"
        "- Prefer a concise direct answer followed by a short supporting explanation.\n\n"
        "Question and evidence:\n"
        f"{json.dumps(context, ensure_ascii=True)}"
    )
