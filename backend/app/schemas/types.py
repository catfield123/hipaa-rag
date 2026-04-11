"""Shared string enums for backend schemas."""

from __future__ import annotations

from enum import StrEnum


class StructuralContentTargetEnum(StrEnum):
    """Targets for rows in ``structural_content`` (section body vs outlines).

    Args (members):
        SECTION_TEXT (str): Full text for one section.
        PART_OUTLINE (str): Part-level outline listing subparts/sections.
        SUBPART_OUTLINE (str): Subpart-level outline listing sections.
    """

    SECTION_TEXT = "section_text"
    PART_OUTLINE = "part_outline"
    SUBPART_OUTLINE = "subpart_outline"


class RetrievalModeEnum(StrEnum):
    """Which backend produced a :class:`~app.schemas.retrieval.RetrievalEvidence` row.

    Args (members):
        BM25_ONLY (str): Lexical BM25 hit.
        DENSE (str): Dense vector similarity hit.
        HYBRID (str): Fused BM25 + dense hit.
        STRUCTURE_LOOKUP (str): Precomputed structural content row.
    """

    BM25_ONLY = "bm25_only"
    DENSE = "dense"
    HYBRID = "hybrid"
    STRUCTURE_LOOKUP = "structure_lookup"


class QueryIntentEnum(StrEnum):
    """Classifier output for answer path selection in :class:`~app.schemas.planning.ResearchDecision`.

    Args (members):
        GENERAL (str): Default Q&A.
        QUOTE_REQUEST (str): User wants verbatim regulatory text.
        EXISTENCE_CHECK (str): Whether a rule/fact appears.
        LIST_REFERENCES (str): Enumerate applicable sections/rules.
        AMBIGUOUS (str): Clarification needed.
        STRUCTURE_LOOKUP (str): User wants outlines or hierarchy.
    """

    GENERAL = "general"
    QUOTE_REQUEST = "quote_request"
    EXISTENCE_CHECK = "existence_check"
    LIST_REFERENCES = "list_references"
    AMBIGUOUS = "ambiguous"
    STRUCTURE_LOOKUP = "structure_lookup"
