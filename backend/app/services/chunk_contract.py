"""Helpers for turning ORM rows into retrieval contracts and SQL filters."""

from __future__ import annotations

import re

from app.models import RetrievalChunk, StructuralContent
from app.schemas.retrieval import RetrievalEvidence, StructuralFilters
from app.schemas.types import RetrievalModeEnum
from sqlalchemy.sql.elements import ColumnElement


def build_retrieval_evidence(
    chunk: RetrievalChunk,
    *,
    retrieval_mode: RetrievalModeEnum,
    score: float,
    metadata_extra: dict[str, object] | None = None,
) -> RetrievalEvidence:
    """Convert a ``RetrievalChunk`` ORM row into a normalized API evidence object.

    Args:
        chunk (RetrievalChunk): Materialized chunk row from the database.
        retrieval_mode (RetrievalModeEnum): Which backend produced this hit.
        score (float): Fusion or similarity score for ranking.
        metadata_extra (dict[str, object] | None): Extra fields merged into ``metadata`` (overrides chunk keys on clash).

    Returns:
        RetrievalEvidence: Pydantic evidence DTO for the answering layer.

    Raises:
        None
    """

    return RetrievalEvidence(
        chunk_id=chunk.id,
        path=[str(value) for value in chunk.path],
        path_text=chunk.path_text,
        text=chunk.text,
        section=chunk.section,
        part=chunk.part,
        subpart=chunk.subpart,
        markers=[str(value) for value in chunk.markers],
        retrieval_mode=retrieval_mode,
        score=score,
        metadata={
            **chunk.metadata_json,
            **(metadata_extra or {}),
        },
    )


def build_structural_content_evidence(content: StructuralContent) -> RetrievalEvidence:
    """Convert a ``StructuralContent`` row into normalized evidence for structure lookups.

    Args:
        content (StructuralContent): Precomputed section or outline row.

    Returns:
        RetrievalEvidence: Evidence with synthetic negative ``chunk_id`` and mode ``structure_lookup``.

    Raises:
        None
    """

    return RetrievalEvidence(
        chunk_id=-content.id,
        path=[str(value) for value in content.path],
        path_text=content.path_text,
        text=content.text,
        section=content.section,
        part=content.part,
        subpart=content.subpart,
        markers=[],
        retrieval_mode=RetrievalModeEnum.STRUCTURE_LOOKUP,
        score=1.0,
        metadata={
            **content.metadata_json,
            "content_type": content.content_type,
            "part_number": content.part_number,
            "subpart_key": content.subpart_key,
            "section_number": content.section_number,
        },
    )


def matches_structural_filters(chunk: RetrievalChunk, filters: StructuralFilters) -> bool:
    """Return whether a chunk row satisfies all non-empty structural filter fields.

    Args:
        chunk (RetrievalChunk): Row to test (in-memory or loaded).
        filters (StructuralFilters): User-provided part/section/subpart/marker constraints.

    Returns:
        bool: ``True`` if every specified filter matches.

    Raises:
        None
    """

    if filters.part_number and not _matches_label_number(chunk.part, "PART", filters.part_number):
        return False

    if filters.section_number and not _matches_label_number(chunk.section, "§", filters.section_number):
        return False

    if filters.subpart and not _matches_label_number(chunk.subpart, "Subpart", filters.subpart):
        return False

    if filters.marker_path:
        expected = [_normalize_marker(marker) for marker in filters.marker_path]
        actual = [_normalize_marker(marker) for marker in chunk.markers]
        if len(actual) < len(expected) or actual[-len(expected) :] != expected:
            return False

    return True


def build_structural_filter_clauses(
    filters: StructuralFilters | None,
    *,
    table: type[RetrievalChunk] = RetrievalChunk,
) -> list[ColumnElement[bool]]:
    """Build SQLAlchemy boolean clauses for ``RetrievalChunk`` filtered search.

    Args:
        filters (StructuralFilters | None): Optional structural constraints; ``None`` yields no clauses.
        table (type[RetrievalChunk]): Mapped class to generate column references (defaults to :class:`RetrievalChunk`).

    Returns:
        list[ColumnElement[bool]]: SQL fragments combined with ``AND`` by callers.

    Raises:
        None
    """

    if filters is None:
        return []

    clauses: list[ColumnElement[bool]] = []
    if filters.part_number:
        clauses.append(table.part.op("~*")(_label_number_pattern("PART", filters.part_number)))

    if filters.section_number:
        clauses.append(table.section.op("~*")(_label_number_pattern("§", filters.section_number)))

    if filters.subpart:
        clauses.append(table.subpart.op("~*")(_label_number_pattern("Subpart", filters.subpart)))

    if filters.marker_path:
        clauses.append(table.markers == [_format_marker(marker) for marker in filters.marker_path])

    return clauses


def build_structural_content_filter_clauses(
    filters: StructuralFilters | None,
    *,
    table: type[StructuralContent] = StructuralContent,
) -> list[ColumnElement[bool]]:
    """Build SQLAlchemy boolean clauses for ``StructuralContent`` filtered lookup.

    Args:
        filters (StructuralFilters | None): Optional constraints; ``None`` yields no clauses.
        table (type[StructuralContent]): Mapped class (defaults to :class:`StructuralContent`).

    Returns:
        list[ColumnElement[bool]]: Equality / key-match clauses for structural rows.

    Raises:
        None
    """

    if filters is None:
        return []

    clauses: list[ColumnElement[bool]] = []
    if filters.part_number:
        clauses.append(table.part_number == filters.part_number)

    if filters.section_number:
        clauses.append(table.section_number == filters.section_number)

    if filters.subpart:
        clauses.append(table.subpart_key == filters.subpart.upper())

    return clauses


def _matches_label_number(label: str | None, prefix: str, expected: str) -> bool:
    """Match a human-readable label (e.g. ``PART 164``) against a numeric or letter expectation.

    Args:
        label (str | None): Column value from the chunk row.
        prefix (str): Label prefix such as ``PART``, ``§``, or ``Subpart``.
        expected (str): User filter fragment (e.g. ``164`` or ``A``).

    Returns:
        bool: ``True`` if the regex built from ``prefix`` and ``expected`` matches ``label``.

    Raises:
        None
    """

    if not label:
        return False

    return bool(re.search(_label_number_pattern(prefix, expected), label, flags=re.IGNORECASE))


def _label_number_pattern(prefix: str, expected: str) -> str:
    """Build a case-insensitive regex fragment for ``prefix`` + ``expected`` at label start.

    Args:
        prefix (str): Literal regulatory prefix.
        expected (str): Expected number or letter token.

    Returns:
        str: Regex pattern string safe for PostgreSQL ``~*``.

    Raises:
        None
    """

    escaped_prefix = re.escape(prefix)
    escaped_expected = re.escape(expected)
    return rf"^{escaped_prefix}\s*{escaped_expected}\b"


def _normalize_marker(marker: str) -> str:
    """Normalize a marker for suffix comparison (strip parentheses, lower-case).

    Args:
        marker (str): Raw marker from user filter or chunk JSON.

    Returns:
        str: Lowercase inner token without parentheses.

    Raises:
        None
    """

    cleaned = marker.strip()
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1]
    return cleaned.lower()


def _format_marker(marker: str) -> str:
    """Ensure a marker string includes parentheses for JSONB array equality checks.

    Args:
        marker (str): Marker from user filter path.

    Returns:
        str: Parenthesized marker as stored in ``RetrievalChunk.markers``.

    Raises:
        None
    """

    normalized = marker.strip()
    if normalized.startswith("(") and normalized.endswith(")"):
        return normalized
    return f"({normalized})"
