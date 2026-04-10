"""Shared literal type aliases for backend schemas."""

from __future__ import annotations

from typing import Literal

StructuralContentTarget = Literal["section_text", "part_outline", "subpart_outline"]
QueryMode = Literal["bm25_only", "hybrid", "structure_lookup"]
RetrievalMode = Literal["bm25_only", "dense", "hybrid", "structure_lookup"]
QueryIntent = Literal[
    "general",
    "quote_request",
    "existence_check",
    "list_references",
    "ambiguous",
    "structure_lookup",
]
