"""Shared string enums for backend schemas."""

from __future__ import annotations

from enum import StrEnum


class StructuralContentTargetEnum(StrEnum):
    """Structural content shapes available through direct lookup."""

    SECTION_TEXT = "section_text"
    PART_OUTLINE = "part_outline"
    SUBPART_OUTLINE = "subpart_outline"


class QueryModeEnum(StrEnum):
    """Retrieval strategies available to a planned query."""

    BM25_ONLY = "bm25_only"
    HYBRID = "hybrid"
    STRUCTURE_LOOKUP = "structure_lookup"


class RetrievalModeEnum(StrEnum):
    """Retrieval backends recorded on evidence items."""

    BM25_ONLY = "bm25_only"
    DENSE = "dense"
    HYBRID = "hybrid"
    STRUCTURE_LOOKUP = "structure_lookup"


class QueryIntentEnum(StrEnum):
    """High-level intents inferred for question answering."""

    GENERAL = "general"
    QUOTE_REQUEST = "quote_request"
    EXISTENCE_CHECK = "existence_check"
    LIST_REFERENCES = "list_references"
    AMBIGUOUS = "ambiguous"
    STRUCTURE_LOOKUP = "structure_lookup"
