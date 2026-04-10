from __future__ import annotations

from collections import defaultdict
import re
from dataclasses import dataclass

from app.services.pdf_parser import ParsedDocument, ParsedNode
from app.services.text_utils import estimate_token_count, normalize_text


@dataclass
class ChunkRecord:
    start_node_key: str
    end_node_key: str
    quote_node_key: str
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


@dataclass
class ChunkAtom:
    node_key: str
    parent_key: str | None
    source_label: str
    text: str
    token_count: int
    page_start: int
    page_end: int
    node_type: str
    marker: str | None


class RetrievalChunker:
    def __init__(
        self,
        min_tokens: int = 30,
        target_tokens: int = 140,
        hard_cap_tokens: int = 220,
    ) -> None:
        self.min_tokens = min_tokens
        self.target_tokens = target_tokens
        self.hard_cap_tokens = hard_cap_tokens

    def build_chunks(self, document: ParsedDocument) -> list[ChunkRecord]:
        nodes_by_key = {node.key: node for node in document.nodes}
        children_by_parent = self._build_children_map(document.nodes)
        section_nodes = [node for node in document.nodes if node.node_type == "section"]
        chunks: list[ChunkRecord] = []
        chunk_counter = 0

        for section in section_nodes:
            atoms = self._build_atoms(section, children_by_parent, nodes_by_key)
            if not atoms:
                continue

            for atom in atoms:
                # Keep chunking aligned to legal structure:
                # one leaf node (paragraph/subparagraph/text) is one base chunk.
                # Only split further when that node is itself too long.
                if atom.token_count > self.target_tokens:
                    for start_char, end_char, text in self._split_large_text(atom.text):
                        content = normalize_text(text)
                        if not content:
                            continue
                        context = self._build_chunk_context(section, nodes_by_key, atom.parent_key)
                        content_with_context = f"{context}\n\n{content}" if context else content
                        chunks.append(
                            ChunkRecord(
                                start_node_key=atom.node_key,
                                end_node_key=atom.node_key,
                                quote_node_key=atom.node_key,
                                chunk_index=chunk_counter,
                                source_label=atom.source_label,
                                content=content,
                                content_with_context=content_with_context,
                                token_count=estimate_token_count(content),
                                char_start=start_char,
                                char_end=end_char,
                                page_start=atom.page_start,
                                page_end=atom.page_end,
                                metadata={
                                    "node_type": atom.node_type,
                                    "marker": atom.marker,
                                    "section_source_label": section.source_label,
                                    "included_node_keys": [atom.node_key],
                                    "included_source_labels": [atom.source_label],
                                },
                            )
                        )
                        chunk_counter += 1
                    continue

                chunks.append(self._build_chunk_record(section, [atom], nodes_by_key, chunk_counter))
                chunk_counter += 1

        return chunks

    def _build_atoms(
        self,
        section: ParsedNode,
        children_by_parent: dict[str, list[str]],
        nodes_by_key: dict[str, ParsedNode],
    ) -> list[ChunkAtom]:
        atoms: list[ChunkAtom] = []
        for leaf in self._collect_leaf_descendants(section.key, children_by_parent, nodes_by_key):
            text = normalize_text(leaf.raw_text)
            if not text:
                continue
            atoms.append(
                ChunkAtom(
                    node_key=leaf.key,
                    parent_key=leaf.parent_key,
                    source_label=leaf.source_label,
                    text=text,
                    token_count=estimate_token_count(text),
                    page_start=leaf.page_start,
                    page_end=leaf.page_end,
                    node_type=leaf.node_type,
                    marker=leaf.marker,
                )
            )
        return atoms

    def _split_large_text(self, text: str) -> list[tuple[int, int, str]]:
        normalized = normalize_text(text)
        if estimate_token_count(normalized) <= self.target_tokens:
            return [(0, len(normalized), normalized)]

        paragraph_like_parts = [part.strip() for part in re.split(r"\n{2,}", normalized) if part.strip()]
        if len(paragraph_like_parts) > 1:
            windows: list[tuple[int, int, str]] = []
            current_parts: list[str] = []
            search_offset = 0
            for part in paragraph_like_parts:
                candidate = "\n\n".join(current_parts + [part]).strip()
                if current_parts and estimate_token_count(candidate) > self.target_tokens:
                    window_text = "\n\n".join(current_parts).strip()
                    start_idx = normalized.find(window_text, search_offset)
                    end_idx = start_idx + len(window_text)
                    windows.append((start_idx, end_idx, window_text))
                    search_offset = max(end_idx - 80, 0)
                    current_parts = [part]
                    continue
                current_parts.append(part)

            if current_parts:
                window_text = "\n\n".join(current_parts).strip()
                start_idx = normalized.find(window_text, search_offset)
                end_idx = start_idx + len(window_text)
                windows.append((start_idx, end_idx, window_text))
            return windows

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

    def _build_children_map(self, nodes: list[ParsedNode]) -> dict[str, list[str]]:
        children_by_parent: dict[str, list[str]] = defaultdict(list)
        for node in nodes:
            if node.parent_key:
                children_by_parent[node.parent_key].append(node.key)
        return children_by_parent

    def _collect_leaf_descendants(
        self,
        node_key: str,
        children_by_parent: dict[str, list[str]],
        nodes_by_key: dict[str, ParsedNode],
    ) -> list[ParsedNode]:
        children = children_by_parent.get(node_key, [])
        if not children:
            return [nodes_by_key[node_key]]

        leaves: list[ParsedNode] = []
        for child_key in children:
            leaves.extend(self._collect_leaf_descendants(child_key, children_by_parent, nodes_by_key))
        return leaves

    def _build_chunk_record(
        self,
        section: ParsedNode,
        atoms: list[ChunkAtom],
        nodes_by_key: dict[str, ParsedNode],
        chunk_index: int,
    ) -> ChunkRecord:
        content = "\n\n".join(atom.text for atom in atoms)
        start_node_key = atoms[0].node_key
        end_node_key = atoms[-1].node_key
        quote_node_key = max(atoms, key=lambda atom: atom.token_count).node_key
        context = self._build_chunk_context(section, nodes_by_key, atoms[0].parent_key)
        content_with_context = f"{context}\n\n{content}" if context else content
        source_label = atoms[0].source_label
        if len(atoms) > 1 and atoms[0].source_label != atoms[-1].source_label:
            source_label = f"{atoms[0].source_label} -> {atoms[-1].source_label}"

        return ChunkRecord(
            start_node_key=start_node_key,
            end_node_key=end_node_key,
            quote_node_key=quote_node_key,
            chunk_index=chunk_index,
            source_label=source_label,
            content=content,
            content_with_context=content_with_context,
            token_count=estimate_token_count(content),
            char_start=0,
            char_end=len(content),
            page_start=min(atom.page_start for atom in atoms),
            page_end=max(atom.page_end for atom in atoms),
            metadata={
                "section_source_label": section.source_label,
                "node_type": atoms[0].node_type if len({atom.node_type for atom in atoms}) == 1 else "mixed",
                "marker": atoms[0].marker if len(atoms) == 1 else None,
                "included_node_keys": [atom.node_key for atom in atoms],
                "included_source_labels": [atom.source_label for atom in atoms],
                "quote_node_key": quote_node_key,
            },
        )

    def _build_chunk_context(
        self,
        section: ParsedNode,
        nodes_by_key: dict[str, ParsedNode],
        parent_key: str | None,
    ) -> str:
        ancestors: list[str] = []
        current = section
        while current.parent_key:
            parent = nodes_by_key.get(current.parent_key)
            if parent is None:
                break
            label = parent.source_label
            if parent.heading:
                label = f"{label} {parent.heading}"
            ancestors.append(label)
            current = parent

        ancestors.reverse()
        section_label = section.source_label
        if section.heading:
            section_label = f"{section_label} {section.heading}"
        parts = ancestors + [section_label]

        parent = nodes_by_key.get(parent_key) if parent_key else None
        if parent and parent.key != section.key and parent.source_label != section.source_label:
            parent_label = parent.source_label
            if parent.heading:
                parent_label = f"{parent_label} {parent.heading}"
            parts.append(parent_label)

        if not parts:
            return ""

        parts = list(dict.fromkeys(parts))
        return " > ".join(parts)

    def _atoms_from_chunk_metadata(
        self,
        chunk: ChunkRecord,
        nodes_by_key: dict[str, ParsedNode],
    ) -> list[ChunkAtom]:
        included_node_keys = chunk.metadata.get("included_node_keys", [])
        atoms: list[ChunkAtom] = []
        for node_key in included_node_keys:
            node = nodes_by_key.get(str(node_key))
            if node is None:
                continue
            text = normalize_text(node.raw_text)
            if not text:
                continue
            atoms.append(
                ChunkAtom(
                    node_key=node.key,
                    parent_key=node.parent_key,
                    source_label=node.source_label,
                    text=text,
                    token_count=estimate_token_count(text),
                    page_start=node.page_start,
                    page_end=node.page_end,
                    node_type=node.node_type,
                    marker=node.marker,
                )
            )
        return atoms
