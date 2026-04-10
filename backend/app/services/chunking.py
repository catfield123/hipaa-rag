from __future__ import annotations

from collections import Counter, defaultdict
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
    section_key: str
    part_number: str | None
    subpart: str | None
    section_number: str | None
    section_source_label: str
    source_label: str
    token: str
    page_start: int
    page_end: int
    marker: str | None


class RetrievalChunker:
    def __init__(
        self,
        window_tokens: int = 64,
        overlap_tokens: int = 20,
    ) -> None:
        self.window_tokens = window_tokens
        self.overlap_tokens = overlap_tokens

    def build_chunks(self, document: ParsedDocument) -> list[ChunkRecord]:
        nodes_by_key = {node.key: node for node in document.nodes}
        children_by_parent = self._build_children_map(document.nodes)
        section_nodes = [node for node in document.nodes if node.node_type == "section"]
        all_atoms: list[ChunkAtom] = []
        chunks: list[ChunkRecord] = []
        chunk_counter = 0

        for section in section_nodes:
            all_atoms.extend(self._build_atoms(section, children_by_parent, nodes_by_key))

        for window_atoms in self._build_windows(all_atoms):
            chunks.append(self._build_chunk_record(window_atoms, nodes_by_key, chunk_counter))
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
            for token in text.split():
                atoms.append(
                    ChunkAtom(
                        node_key=leaf.key,
                        parent_key=leaf.parent_key,
                        section_key=section.key,
                        part_number=leaf.part_number,
                        subpart=leaf.subpart,
                        section_number=leaf.section_number,
                        section_source_label=section.source_label,
                        source_label=leaf.source_label,
                        token=token,
                        page_start=leaf.page_start,
                        page_end=leaf.page_end,
                        marker=leaf.marker,
                    )
                )
        return atoms

    def _build_windows(self, atoms: list[ChunkAtom]) -> list[list[ChunkAtom]]:
        if not atoms:
            return []

        windows: list[list[ChunkAtom]] = []
        step = max(1, self.window_tokens - self.overlap_tokens)

        for start_index in range(0, len(atoms), step):
            window_atoms = atoms[start_index : start_index + self.window_tokens]
            if not window_atoms:
                continue
            windows.append(window_atoms)
            if start_index + self.window_tokens >= len(atoms):
                break

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
        atoms: list[ChunkAtom],
        nodes_by_key: dict[str, ParsedNode],
        chunk_index: int,
    ) -> ChunkRecord:
        content = " ".join(atom.token for atom in atoms).strip()
        start_node_key = atoms[0].node_key
        end_node_key = atoms[-1].node_key
        start_section = nodes_by_key[atoms[0].section_key]
        end_section = nodes_by_key[atoms[-1].section_key]
        quote_node_key = Counter(atom.node_key for atom in atoms).most_common(1)[0][0]
        content_with_context = content
        # Keep the persisted chunk label short and stable for DB storage,
        # while the full structural context stays in metadata/content_with_context.
        source_label = f"pp. {min(atom.page_start for atom in atoms)}-{max(atom.page_end for atom in atoms)}"
        if start_section.key != end_section.key:
            source_label = (
                f"pp. {min(atom.page_start for atom in atoms)}-{max(atom.page_end for atom in atoms)} "
                f"({start_section.source_label} -> {end_section.source_label})"
            )[:255]
        markers = list(dict.fromkeys(atom.marker for atom in atoms if atom.marker))
        section_keys = list(dict.fromkeys(atom.section_key for atom in atoms))
        section_labels = [
            nodes_by_key[section_key].source_label
            for section_key in section_keys
            if section_key in nodes_by_key
        ]
        section_numbers = list(dict.fromkeys(atom.section_number for atom in atoms if atom.section_number))
        part_numbers = list(dict.fromkeys(atom.part_number for atom in atoms if atom.part_number))
        subparts = list(dict.fromkeys(atom.subpart for atom in atoms if atom.subpart))
        node_keys = [atom.node_key for atom in atoms]
        node_labels = [atom.source_label for atom in atoms]

        return ChunkRecord(
            start_node_key=start_node_key,
            end_node_key=end_node_key,
            quote_node_key=quote_node_key,
            chunk_index=chunk_index,
            source_label=source_label,
            content=content,
            content_with_context=content_with_context,
            token_count=len(atoms),
            char_start=0,
            char_end=len(normalize_text(nodes_by_key[quote_node_key].raw_text)),
            page_start=min(atom.page_start for atom in atoms),
            page_end=max(atom.page_end for atom in atoms),
            metadata={
                "part_number": start_section.part_number,
                "subpart": start_section.subpart,
                "section_number": start_section.section_number,
                "section_node_key": start_section.key,
                "section_source_label": start_section.source_label,
                "section_heading": start_section.heading,
                "start_section_node_key": start_section.key,
                "end_section_node_key": end_section.key,
                "included_section_keys": section_keys,
                "included_section_labels": section_labels,
                "included_section_numbers": section_numbers,
                "included_section_count": len(section_keys),
                "included_part_numbers": part_numbers,
                "included_subparts": subparts,
                "marker": atoms[0].marker if len(markers) == 1 else None,
                "included_markers": markers,
                "included_node_keys": node_keys,
                "included_source_labels": node_labels,
                "included_node_count": len(node_keys),
                "start_source_label": atoms[0].source_label,
                "end_source_label": atoms[-1].source_label,
                "quote_node_key": quote_node_key,
                "crosses_node_boundary": len(node_keys) > 1,
                "crosses_section_boundary": len(section_keys) > 1,
                "window_token_target": self.window_tokens,
                "window_token_overlap": self.overlap_tokens,
            },
        )
