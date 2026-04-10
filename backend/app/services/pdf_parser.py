from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pypdf import PdfReader

from app.services.text_utils import marker_depth, normalize_line, normalize_text


SECTION_PATTERN = re.compile(r"^§\s*(\d+\.\d+)\s*(.*)$")
PART_PATTERN = re.compile(r"^PART\s+(\d+)[—-](.+)$")
SUBPART_PATTERN = re.compile(r"^SUBPART\s+([A-Z])[—-](.+)$")
MARKER_PATTERN = re.compile(r"^\(([A-Za-z0-9ivxlcdmIVXLCDM]+)\)\s*(.*)$")


@dataclass
class LineEntry:
    text: str
    page_number: int


@dataclass
class ParsedNode:
    key: str
    parent_key: str | None
    node_type: Literal["part", "subpart", "section", "paragraph", "subparagraph", "text"]
    part_number: str | None
    subpart: str | None
    section_number: str | None
    marker: str | None
    source_label: str
    heading: str | None
    raw_text: str
    page_start: int
    page_end: int


@dataclass
class ParsedSection:
    part_number: str | None
    subpart: str | None
    section_number: str
    heading: str
    page_start: int
    page_end: int
    lines: list[LineEntry] = field(default_factory=list)


@dataclass
class ParsedDocument:
    nodes: list[ParsedNode]


class HIPAAPdfParser:
    def __init__(self) -> None:
        self._node_counter = 0

    def parse(self, pdf_path: str) -> ParsedDocument:
        pages = self._extract_clean_pages(pdf_path)
        nodes: list[ParsedNode] = []

        current_part_key: str | None = None
        current_subpart_key: str | None = None
        current_part_number: str | None = None
        current_subpart: str | None = None
        current_section: ParsedSection | None = None

        for page_number, lines in pages:
            i = 0
            while i < len(lines):
                line = lines[i]
                part_match = PART_PATTERN.match(line)
                if part_match:
                    if current_section is not None:
                        nodes.extend(self._flush_section(current_section, current_subpart_key or current_part_key))
                        current_section = None

                    current_part_number = part_match.group(1)
                    current_subpart = None
                    part_heading = part_match.group(2).strip()
                    current_part_key = self._next_key()
                    nodes.append(
                        ParsedNode(
                            key=current_part_key,
                            parent_key=None,
                            node_type="part",
                            part_number=current_part_number,
                            subpart=None,
                            section_number=None,
                            marker=None,
                            source_label=f"45 CFR Part {current_part_number}",
                            heading=part_heading,
                            raw_text=part_heading,
                            page_start=page_number,
                            page_end=page_number,
                        )
                    )
                    current_subpart_key = None
                    i += 1
                    continue

                subpart_match = SUBPART_PATTERN.match(line)
                if subpart_match:
                    if current_section is not None:
                        nodes.extend(self._flush_section(current_section, current_subpart_key or current_part_key))
                        current_section = None

                    current_subpart = subpart_match.group(1)
                    subpart_heading = subpart_match.group(2).strip()
                    current_subpart_key = self._next_key()
                    nodes.append(
                        ParsedNode(
                            key=current_subpart_key,
                            parent_key=current_part_key,
                            node_type="subpart",
                            part_number=current_part_number,
                            subpart=current_subpart,
                            section_number=None,
                            marker=None,
                            source_label=f"45 CFR Part {current_part_number} Subpart {current_subpart}",
                            heading=subpart_heading,
                            raw_text=subpart_heading,
                            page_start=page_number,
                            page_end=page_number,
                        )
                    )
                    i += 1
                    continue

                section_match = SECTION_PATTERN.match(line)
                if section_match:
                    if current_section is not None:
                        nodes.extend(self._flush_section(current_section, current_subpart_key or current_part_key))

                    section_number = section_match.group(1)
                    section_heading, inline_body = self._split_section_heading_and_inline_body(
                        section_match.group(2).strip()
                    )
                    heading_lines = [section_heading] if section_heading else []
                    section_lines: list[LineEntry] = []
                    if inline_body:
                        section_lines.append(LineEntry(text=inline_body, page_number=page_number))
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j]
                        if PART_PATTERN.match(next_line) or SUBPART_PATTERN.match(next_line) or SECTION_PATTERN.match(next_line):
                            break
                        if MARKER_PATTERN.match(next_line):
                            break
                        heading_lines.append(next_line)
                        if next_line.endswith("."):
                            j += 1
                            break
                        j += 1

                    current_section = ParsedSection(
                        part_number=current_part_number,
                        subpart=current_subpart,
                        section_number=section_number,
                        heading=normalize_text(" ".join(heading_lines)),
                        page_start=page_number,
                        page_end=page_number,
                        lines=section_lines,
                    )
                    i = j
                    continue

                if current_section is not None:
                    current_section.lines.append(LineEntry(text=line, page_number=page_number))
                    current_section.page_end = page_number

                i += 1

        if current_section is not None:
            nodes.extend(self._flush_section(current_section, current_subpart_key or current_part_key))

        return ParsedDocument(nodes=nodes)

    def _flush_section(self, section: ParsedSection, parent_key: str | None) -> list[ParsedNode]:
        section_key = self._next_key()
        section_label = f"45 CFR § {section.section_number}"
        section_body = normalize_text("\n".join(line.text for line in section.lines))
        section_node = ParsedNode(
            key=section_key,
            parent_key=parent_key,
            node_type="section",
            part_number=section.part_number,
            subpart=section.subpart,
            section_number=section.section_number,
            marker=None,
            source_label=section_label,
            heading=section.heading,
            raw_text=section_body or section.heading,
            page_start=section.page_start,
            page_end=section.page_end,
        )

        nodes = [section_node]
        logical_blocks = self._build_logical_blocks(section.lines)
        if not logical_blocks:
            return nodes

        stack: list[tuple[int, ParsedNode]] = [(0, section_node)]
        for block in logical_blocks:
            if block["marker"] is None:
                raw_text = block["text"]
                if not raw_text:
                    continue
                nodes.append(
                    ParsedNode(
                        key=self._next_key(),
                        parent_key=section_key,
                        node_type="text",
                        part_number=section.part_number,
                        subpart=section.subpart,
                        section_number=section.section_number,
                        marker=None,
                        source_label=section_label,
                        heading=None,
                        raw_text=raw_text,
                        page_start=block["page_start"],
                        page_end=block["page_end"],
                    )
                )
                continue

            depth = marker_depth(block["marker"])
            while stack and stack[-1][0] >= depth:
                stack.pop()

            parent = stack[-1][1] if stack else section_node
            node_type = "paragraph" if depth <= 2 else "subparagraph"
            source_label = f"{parent.source_label}{block['marker']}"
            node = ParsedNode(
                key=self._next_key(),
                parent_key=parent.key,
                node_type=node_type,
                part_number=section.part_number,
                subpart=section.subpart,
                section_number=section.section_number,
                marker=block["marker"],
                source_label=source_label,
                heading=None,
                raw_text=block["text"],
                page_start=block["page_start"],
                page_end=block["page_end"],
            )
            nodes.append(node)
            stack.append((depth, node))

        return nodes

    def _build_logical_blocks(self, lines: list[LineEntry]) -> list[dict[str, int | str | None]]:
        blocks: list[dict[str, int | str | None]] = []
        current_marker: str | None = None
        current_lines: list[str] = []
        page_start: int | None = None
        page_end: int | None = None

        def flush() -> None:
            nonlocal current_marker, current_lines, page_start, page_end
            text = normalize_text(" ".join(current_lines))
            if text and page_start is not None and page_end is not None:
                blocks.append(
                    {
                        "marker": current_marker,
                        "text": text,
                        "page_start": page_start,
                        "page_end": page_end,
                    }
                )
            current_marker = None
            current_lines = []
            page_start = None
            page_end = None

        for line in lines:
            marker_match = MARKER_PATTERN.match(line.text)
            if marker_match:
                flush()
                current_marker = f"({marker_match.group(1)})"
                current_lines = [marker_match.group(2).strip()]
                page_start = line.page_number
                page_end = line.page_number
                continue

            if page_start is None:
                page_start = line.page_number
            page_end = line.page_number
            current_lines.append(line.text)

        flush()
        return blocks

    def _split_section_heading_and_inline_body(self, text: str) -> tuple[str, str | None]:
        normalized = normalize_text(text)
        if not normalized:
            return "", None

        if ". " not in normalized:
            return normalized, None

        heading_candidate, remainder = normalized.split(". ", 1)
        heading_candidate = f"{heading_candidate.strip()}."
        remainder = remainder.strip()

        if (
            3 <= len(heading_candidate) <= 220
            and remainder
            and (remainder[0].isalnum() or remainder[0] == "(")
        ):
            return heading_candidate, remainder

        return normalized, None

    def _extract_clean_pages(self, pdf_path: str) -> list[tuple[int, list[str]]]:
        reader = PdfReader(pdf_path)
        pages: list[tuple[int, list[str]]] = []
        for index, page in enumerate(reader.pages, start=1):
            if index <= 8:
                continue

            text = page.extract_text() or ""
            cleaned_lines = self._clean_page_lines(text)
            if cleaned_lines:
                pages.append((index, cleaned_lines))

        return pages

    def _clean_page_lines(self, text: str) -> list[str]:
        raw_lines = [normalize_line(line) for line in text.splitlines()]
        cleaned: list[str] = []
        for line in raw_lines:
            if not line:
                continue
            if line.startswith("-- ") and " of 115 " in line:
                continue
            if line == "HIPAA Administrative Simplification Regulation Text":
                continue
            if line == "March 2013":
                continue
            if line.isdigit():
                continue
            if line == "Contents":
                continue
            if Path(line).name == line and line.endswith(".pdf"):
                continue
            cleaned.append(line)
        return cleaned

    def _next_key(self) -> str:
        self._node_counter += 1
        return f"node-{self._node_counter}"
