from __future__ import annotations

import re

from sqlalchemy.sql.elements import ColumnElement

from app.models import RetrievalChunk
from app.schemas import RetrievalEvidence, StructuralFilters


def build_retrieval_evidence(
    chunk: RetrievalChunk,
    *,
    retrieval_mode: str,
    score: float,
    metadata_extra: dict[str, object] | None = None,
) -> RetrievalEvidence:
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


def matches_structural_filters(chunk: RetrievalChunk, filters: StructuralFilters) -> bool:
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


def _matches_label_number(label: str | None, prefix: str, expected: str) -> bool:
    if not label:
        return False

    return bool(re.search(_label_number_pattern(prefix, expected), label, flags=re.IGNORECASE))


def _label_number_pattern(prefix: str, expected: str) -> str:
    escaped_prefix = re.escape(prefix)
    escaped_expected = re.escape(expected)
    return rf"^{escaped_prefix}\s*{escaped_expected}\b"


def _normalize_marker(marker: str) -> str:
    cleaned = marker.strip()
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1]
    return cleaned.lower()


def _format_marker(marker: str) -> str:
    normalized = marker.strip()
    if normalized.startswith("(") and normalized.endswith(")"):
        return normalized
    return f"({normalized})"
