from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.pdf_parser import ParsedDocument, ParsedNode
from app.services.text_utils import estimate_token_count, normalize_text


@dataclass
class ChunkRecord:
    start_node_key: str
    end_node_key: str
    chunk_index: int
    source_label: str
    content: str
    content_with_context: str
    token_count: int
    char_start: int
    char_end: int
    page_start: int
    page_end: int
    metadata: dict[str, object]


class RetrievalChunker:
    def __init__(self, target_tokens: int = 450, hard_cap_tokens: int = 850) -> None:
        self.target_tokens = target_tokens
        self.hard_cap_tokens = hard_cap_tokens

    def build_chunks(self, document: ParsedDocument) -> list[ChunkRecord]:
        nodes_by_key = {node.key: node for node in document.nodes}
        parent_keys = {node.parent_key for node in document.nodes if node.parent_key}
        leaf_nodes = [node for node in document.nodes if node.key not in parent_keys]
        chunks: list[ChunkRecord] = []

        for node in leaf_nodes:
            for chunk_index, (start_char, end_char, text) in enumerate(
                self._split_node_text(node.raw_text)
            ):
                content = normalize_text(text)
                if not content:
                    continue
                context = self._build_context(nodes_by_key, node)
                content_with_context = f"{context}\n\n{content}" if context else content
                chunks.append(
                    ChunkRecord(
                        start_node_key=node.key,
                        end_node_key=node.key,
                        chunk_index=chunk_index,
                        source_label=node.source_label,
                        content=content,
                        content_with_context=content_with_context,
                        token_count=estimate_token_count(content),
                        char_start=start_char,
                        char_end=end_char,
                        page_start=node.page_start,
                        page_end=node.page_end,
                        metadata={
                            "node_type": node.node_type,
                            "heading": node.heading,
                            "marker": node.marker,
                        },
                    )
                )

        return chunks

    def _split_node_text(self, text: str) -> list[tuple[int, int, str]]:
        normalized = normalize_text(text)
        if estimate_token_count(normalized) <= self.hard_cap_tokens:
            return [(0, len(normalized), normalized)]

        sentences = re.split(r"(?<=[.!?;])\s+", normalized)
        windows: list[tuple[int, int, str]] = []
        current_sentences: list[str] = []
        current_start = 0
        search_offset = 0

        for sentence in sentences:
            candidate = " ".join(current_sentences + [sentence]).strip()
            if current_sentences and estimate_token_count(candidate) > self.target_tokens:
                window_text = " ".join(current_sentences).strip()
                start_idx = normalized.find(window_text, search_offset)
                end_idx = start_idx + len(window_text)
                windows.append((start_idx, end_idx, window_text))
                search_offset = max(end_idx - 80, 0)
                current_sentences = [sentence]
                current_start = search_offset
                continue

            current_sentences.append(sentence)

        if current_sentences:
            window_text = " ".join(current_sentences).strip()
            start_idx = normalized.find(window_text, search_offset)
            if start_idx < 0:
                start_idx = current_start
            end_idx = start_idx + len(window_text)
            windows.append((start_idx, end_idx, window_text))

        return windows

    def _build_context(self, nodes_by_key: dict[str, ParsedNode], node: ParsedNode) -> str:
        parts: list[str] = []
        current = node
        while current.parent_key:
            parent = nodes_by_key.get(current.parent_key)
            if parent is None:
                break
            label = parent.source_label
            if parent.heading:
                label = f"{label} {parent.heading}"
            parts.append(label)
            current = parent

        if not parts:
            return ""

        parts.reverse()
        return " > ".join(parts)
