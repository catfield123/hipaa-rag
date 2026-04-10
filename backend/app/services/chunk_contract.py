from __future__ import annotations

import re

from app.models import RetrievalChunk
from app.schemas import RetrievalEvidence, StructuralFilters


def build_retrieval_evidence(
    chunk: RetrievalChunk,
    *,
    retrieval_mode: str,
    score: float,
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
        metadata=chunk.metadata_json,
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


def _matches_label_number(label: str | None, prefix: str, expected: str) -> bool:
    if not label:
        return False

    escaped_prefix = re.escape(prefix)
    escaped_expected = re.escape(expected)
    pattern = rf"^{escaped_prefix}\s*{escaped_expected}\b"
    return bool(re.search(pattern, label, flags=re.IGNORECASE))


def _normalize_marker(marker: str) -> str:
    cleaned = marker.strip()
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1]
    return cleaned.lower()
