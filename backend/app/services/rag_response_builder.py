"""Helpers for building the public RAG response payload."""

from __future__ import annotations

from app.schemas.chat import QuoteSpan, SourceItem
from app.schemas.retrieval import RetrievalEvidence


class RagResponseBuilder:
    """Build compact quotes and sources from retrieval evidence."""

    def build_quotes(self, evidence: list[RetrievalEvidence]) -> list[QuoteSpan]:
        """Return up to three unique quote spans from the ranked evidence."""

        quotes: list[QuoteSpan] = []
        seen_chunk_ids: set[int] = set()
        for item in evidence[:3]:
            if item.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(item.chunk_id)
            quotes.append(
                QuoteSpan(
                    chunk_id=item.chunk_id,
                    path=item.path,
                    path_text=item.path_text,
                    section=item.section,
                    part=item.part,
                    subpart=item.subpart,
                    markers=item.markers,
                    text=item.text,
                )
            )
        return quotes

    def build_sources(self, evidence: list[RetrievalEvidence]) -> list[SourceItem]:
        """Return up to five unique source entries from the ranked evidence."""

        sources_map: dict[str, SourceItem] = {}
        for item in evidence[:5]:
            sources_map.setdefault(
                item.path_text,
                SourceItem(
                    chunk_id=item.chunk_id,
                    path_text=item.path_text,
                    section=item.section,
                    part=item.part,
                    subpart=item.subpart,
                    markers=item.markers,
                ),
            )
        return list(sources_map.values())
