"""User-role message bodies for OpenAI chat (filled in ``prompts`` builders)."""

from __future__ import annotations

RETRIEVAL_ROUND = (
    "Retrieval round: {round_number}\n\n"
    "Question:\n{question}\n\n"
    "Query budget for this round:\n"
    "- Broad evidence gathering: between {broad_query_min} and {max_queries} retrieval calls.\n"
    "- Narrow follow-up may use one or two calls only if the current evidence already points to one specific "
    "section, part, or subpart.\n\n"
    "Evidence already collected:\n"
    "{evidence_json}\n\n"
    "Previous retrieval calls (full history):\n"
    "{retrieval_history_json}\n\n"
    "Previous decision:\n"
    "{prior_decision_json}"
)

RESEARCH_DECISION = (
    "Decision round after retrieval round: {round_number}\n\n"
    "Question:\n{question}\n\n"
    "Supported sections explicitly present in evidence metadata or text:\n"
    "{supported_sections_json}\n\n"
    "Current evidence:\n"
    "{evidence_json}"
)

FINAL_ANSWER = (
    "Question:\n{question}\n\n"
    "Decision:\n"
    "{decision_json}\n\n"
    "Section labels present in the excerpts below:\n"
    "{supported_sections_json}\n\n"
    "Regulatory text excerpts (JSON):\n"
    "{evidence_json}"
)

FINAL_ANSWER_SYSTEM_WITH_QUOTE_SUPPLEMENT = "{system_prompt}\n\n{quote_supplement}"
