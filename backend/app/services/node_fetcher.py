from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DocumentNode
from app.schemas import NodeResponse, QuoteSpan


class NodeFetcher:
    async def get_node(self, session: AsyncSession, node_id: int) -> NodeResponse:
        node = await session.scalar(select(DocumentNode).where(DocumentNode.id == node_id))
        if node is None:
            raise ValueError(f"Node {node_id} was not found.")

        return NodeResponse(
            id=node.id,
            parent_id=node.parent_id,
            source_label=node.source_label,
            heading=node.heading,
            raw_text=node.raw_text,
            page_start=node.page_start,
            page_end=node.page_end,
        )

    async def get_span(
        self,
        session: AsyncSession,
        node_id: int,
        char_start: int = 0,
        char_end: int | None = None,
        expand: str = "none",
    ) -> QuoteSpan:
        node = await session.scalar(select(DocumentNode).where(DocumentNode.id == node_id))
        if node is None:
            raise ValueError(f"Node {node_id} was not found.")

        text = node.raw_text
        safe_start = max(0, min(char_start, len(text)))
        safe_end = len(text) if char_end is None else max(safe_start, min(char_end, len(text)))

        if expand == "sentence":
            safe_start, safe_end = self._expand_to_sentence(text, safe_start, safe_end)
        elif expand == "paragraph":
            safe_start, safe_end = 0, len(text)

        return QuoteSpan(
            node_id=node.id,
            source_label=node.source_label,
            page_start=node.page_start,
            page_end=node.page_end,
            text=text[safe_start:safe_end].strip(),
            char_start=safe_start,
            char_end=safe_end,
        )

    def _expand_to_sentence(self, text: str, char_start: int, char_end: int) -> tuple[int, int]:
        start = max(text.rfind(". ", 0, char_start), text.rfind("; ", 0, char_start))
        end_candidates = [idx for idx in (text.find(". ", char_end), text.find("; ", char_end)) if idx != -1]
        end = min(end_candidates) if end_candidates else len(text)
        start = 0 if start == -1 else start + 2
        end = len(text) if end == -1 else end + 1

        if not text[start:end].strip():
            match = re.search(r"[^.?!;]+", text)
            if match:
                return match.start(), match.end()

        return start, end
