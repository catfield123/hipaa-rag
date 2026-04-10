"""Function-calling schemas for retrieval continuation decisions."""

from __future__ import annotations

import json
from typing import Any

DECIDE_RESEARCH_STATUS_FUNCTION_NAME = "decide_research_status"


def build_research_decision_functions() -> list[dict[str, Any]]:
    """Return function schemas for the post-retrieval decision step."""

    return [
        {
            "type": "function",
            "function": {
                "name": DECIDE_RESEARCH_STATUS_FUNCTION_NAME,
                "description": (
                    "Decide whether the current evidence is sufficient to answer the HIPAA question "
                    "or whether another retrieval round is required."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": "string",
                            "enum": [
                                "general",
                                "quote_request",
                                "existence_check",
                                "list_references",
                                "ambiguous",
                                "structure_lookup",
                            ],
                        },
                        "wants_raw_structure": {"type": "boolean"},
                        "continue_retrieval": {"type": "boolean"},
                        "rationale": {"type": "string"},
                        "missing_information": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "intent",
                        "wants_raw_structure",
                        "continue_retrieval",
                        "rationale",
                        "missing_information",
                    ],
                    "additionalProperties": False,
                },
            },
        }
    ]


def extract_research_decision_payload(message: Any) -> dict[str, Any]:
    """Extract decision arguments from a model function call."""

    function_calls = getattr(message, "tool_calls", None) or []
    for function_call in function_calls:
        function = getattr(function_call, "function", None)
        if function is None or function.name != DECIDE_RESEARCH_STATUS_FUNCTION_NAME:
            continue
        return json.loads(function.arguments or "{}")
    raise ValueError("Research decision function call missing.")
