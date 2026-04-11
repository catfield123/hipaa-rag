"""Helpers for building the public RAG response payload."""

from __future__ import annotations

from app.schemas.chat import QuoteSpan, SourceItem
from app.schemas.retrieval import RetrievalEvidence


class RagResponseBuilder:
    """Build quotes and sources from full ranked retrieval evidence."""

    def build_quotes(self, evidence: list[RetrievalEvidence]) -> list[QuoteSpan]:
        """Return one quote span per unique chunk, in evidence order.

        Args:
            evidence (list[RetrievalEvidence]): Merged retrieval hits (may contain duplicate chunk ids).

        Returns:
            list[QuoteSpan]: Deduplicated quote rows preserving first-seen order.

        Raises:
            None
        """

        quotes: list[QuoteSpan] = []
        seen_chunk_ids: set[int] = set()
        for item in evidence:
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
        """Return one source row per unique chunk, aligned with ``build_quotes``.

        Args:
            evidence (list[RetrievalEvidence]): Same evidence list used for quotes.

        Returns:
            list[SourceItem]: Deduplicated source rows aligned with ``build_quotes`` chunk order.

        Raises:
            None
        """

        sources: list[SourceItem] = []
        seen_chunk_ids: set[int] = set()
        for item in evidence:
            if item.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(item.chunk_id)
            sources.append(
                SourceItem(
                    chunk_id=item.chunk_id,
                    path_text=item.path_text,
                    section=item.section,
                    part=item.part,
                    subpart=item.subpart,
                    markers=item.markers,
                )
            )
        return sources
