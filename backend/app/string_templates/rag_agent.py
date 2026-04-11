"""Strings for the RAG agent path: WebSocket status lines, tool labels, answering configuration messages."""

from __future__ import annotations

# --- WebSocket / streaming status ---

AGENT_STATUS_PREPARING = "Preparing the agent and HIPAA knowledge-base search…"
AGENT_STATUS_ROUND_PLAN = "Round {round_number} of {max_rounds}: choosing retrieval tool queries…"
AGENT_STATUS_RETRIEVING = "Round {round_number}: retrieving — {tool_label}…"
AGENT_STATUS_DECIDE_CHECK = "Round {round_number}: checking whether evidence is sufficient…"
AGENT_STATUS_GENERATING_FINAL = "Generating the final answer from retrieved sources…"
AGENT_STATUS_MORE_SOURCES = "Round {round_number}: more sources needed. Rationale: {rationale}"
AGENT_STATUS_LIMIT_REACHED = (
    "Retrieval round limit reached ({max_rounds}). " "Generating an answer from available sources…"
)

AGENT_TOOL_LABELS: dict[str, str] = {
    "hybrid_search": "hybrid search",
    "bm25_search": "keyword match (BM25)",
    "lookup_structural_content": "structural content",
    "get_section_text": "section text",
    "list_part_outline": "part outline",
    "list_subpart_outline": "subpart outline",
}

# --- Configuration, runtime, and model fallbacks (answering service) ---

CONFIG_OPENAI_KEY_REQUIRED_FOR_AGENT = (
    "OPENAI_API_KEY is required because answering only supports the function-calling agent."
)
RUNTIME_RETRIEVAL_ROUND_REQUIRES_TOOLS = "Each retrieval round must contain at least one retrieval function call."
FINAL_ANSWER_EMPTY_FALLBACK = "I could not produce a final answer from the retrieved evidence."
RESEARCH_DECISION_RATIONALE_MAX_ROUNDS = "Maximum retrieval rounds reached."
