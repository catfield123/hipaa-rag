"""Pure helpers for the function-calling retrieval agent (evidence merge, tool-call dedupe, history rows)."""

from __future__ import annotations

import json
from typing import Any

from app.schemas.retrieval import RetrievalEvidence
from app.schemas.types import RetrievalCallSkipReason


def truncate_status_line(text: str, *, limit: int = 280) -> str:
    """Collapse whitespace and truncate for compact UI status lines (e.g. decision rationale)."""

    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "…"


def merge_evidence_by_chunk_id(
    existing: list[RetrievalEvidence],
    new_items: list[RetrievalEvidence],
) -> list[RetrievalEvidence]:
    """Merge evidence lists, deduplicating by ``chunk_id`` while preserving first-seen order."""

    merged: list[RetrievalEvidence] = []
    seen_chunk_ids: set[int] = set()
    for item in [*existing, *new_items]:
        if item.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(item.chunk_id)
        merged.append(item)
    return merged


def prepare_retrieval_tool_calls(
    function_calls: list[Any],
    *,
    max_queries: int,
) -> tuple[list[Any], list[dict[str, object]]]:
    """Deduplicate tool calls by normalized arguments and enforce a per-round call budget."""

    prepared_calls: list[Any] = []
    skipped_calls: list[dict[str, object]] = []
    seen_keys: set[str] = set()

    for function_call in function_calls:
        function_name = function_call.function.name
        parsed_arguments = parse_tool_call_json_arguments(function_call.function.arguments)
        call_key = build_retrieval_tool_call_dedupe_key(
            function_name=function_name,
            function_args=parsed_arguments,
        )
        if call_key in seen_keys:
            skipped_calls.append(
                {
                    "function_name": function_name,
                    "function_args": parsed_arguments,
                    "reason": RetrievalCallSkipReason.DUPLICATE_EXACT_CALL.value,
                }
            )
            continue
        if len(prepared_calls) >= max_queries:
            skipped_calls.append(
                {
                    "function_name": function_name,
                    "function_args": parsed_arguments,
                    "reason": RetrievalCallSkipReason.OVER_QUERY_BUDGET.value,
                }
            )
            continue
        seen_keys.add(call_key)
        prepared_calls.append(function_call)

    return prepared_calls, skipped_calls


def build_retrieval_tool_call_dedupe_key(*, function_name: str, function_args: dict[str, Any]) -> str:
    """Return a stable JSON key for deduplicating identical retrieval calls."""

    normalized_args = normalize_tool_arguments_tree(function_args)
    return json.dumps(
        {"function_name": function_name, "function_args": normalized_args},
        ensure_ascii=True,
        sort_keys=True,
    )


def build_retrieval_history_entry(
    *,
    round_number: int,
    function_name: str,
    function_args: dict[str, Any],
    result_count: int,
    error: str | None = None,
) -> dict[str, object]:
    """Serialize one retrieval tool invocation for the next round's user prompt."""

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


def parse_tool_call_json_arguments(raw_arguments: str) -> dict[str, Any]:
    """Parse tool-call JSON; on failure return a dict capturing the raw string."""

    try:
        parsed = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return {"_raw_arguments": raw_arguments}
    return parsed if isinstance(parsed, dict) else {"_raw_arguments": raw_arguments}


def normalize_tool_arguments_tree(value: Any) -> Any:
    """Recursively normalize dict/list/string values for stable dedupe keys."""

    if isinstance(value, dict):
        return {
            str(key): normalize_tool_arguments_tree(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        return [normalize_tool_arguments_tree(item) for item in value]
    if isinstance(value, str):
        normalized = " ".join(value.split())
        return normalized.casefold() if normalized else normalized
    return value
