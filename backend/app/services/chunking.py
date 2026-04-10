from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class MarkdownChunker:
    PART_RE = re.compile(r"^PART\s+(\d{3})(?:\s*[-\s]?\s*(.*))?$", re.IGNORECASE)
    SUBPART_RE = re.compile(r"^Subpart\s+([A-Z])(?:[-\s]+)(.*)")
    SECTION_RE = re.compile(r"^§\s*(\d+\.\d+)\b\s+([A-Z].*)")
    COMPOUND_MARKERS_RE = re.compile(r"^((?:\([A-Za-z0-9ivxlcdmIVXLCDM]+\))+)\s*(.*)$")
    SINGLE_MARKER_RE = re.compile(r"\(([A-Za-z0-9ivxlcdmIVXLCDM]+)\)")
    FR_RE = re.compile(r"^\[\d{2,}\s+FR.*\]$")

    def chunk_markdown(self, src: str | list[str]) -> list[dict[str, Any]]:
        lines = self._normalize_lines(src)
        chunks: list[dict[str, Any]] = []

        current_part_label: str | None = None
        current_subpart_label: str | None = None
        current_section_label: str | None = None
        marker_stack: list[dict[str, Any]] = []
        current_chunk: dict[str, Any] | None = None

        def push_chunk(text: str, markers: list[dict[str, Any]]) -> dict[str, Any]:
            nonlocal current_chunk
            if current_section_label is None:
                raise ValueError("Found content before first section header")

            chunk = self._make_chunk(
                text=text,
                part_label=current_part_label,
                subpart_label=current_subpart_label,
                section_label=current_section_label,
                markers=markers,
            )
            chunks.append(chunk)
            current_chunk = chunk
            return chunk

        for raw in lines:
            line = raw.strip()
            if not line or line == "Contents" or self.FR_RE.fullmatch(line) or "[Reserved]" in line:
                continue

            part_match = self.PART_RE.match(line)
            if part_match:
                part_no, part_title = part_match.groups()
                part_title = (part_title or "").strip()
                current_part_label = f"PART {part_no} {part_title}".strip()
                current_subpart_label = None
                current_section_label = None
                marker_stack = []
                current_chunk = None
                continue

            subpart_match = self.SUBPART_RE.match(line)
            if subpart_match:
                subpart_letter, subpart_title = subpart_match.groups()
                current_subpart_label = (
                    f"Subpart {subpart_letter} - {subpart_title.strip()}".strip(" -")
                )
                current_section_label = None
                marker_stack = []
                current_chunk = None
                continue

            if line.startswith("§ ") and not self.SECTION_RE.match(line):
                if current_chunk is not None:
                    current_chunk["text"] += " " + line
                continue

            section_match = self.SECTION_RE.match(line)
            if section_match:
                sec_no, sec_title = section_match.groups()
                current_section_label = f"§ {sec_no} {sec_title.strip()}".strip()
                marker_stack = []
                current_chunk = None
                continue

            marker_match = self.COMPOUND_MARKERS_RE.match(line)
            if marker_match:
                compound, rest = marker_match.groups()
                tokens = self.SINGLE_MARKER_RE.findall(compound)

                if len(tokens) > 1:
                    marker_stack = [
                        {
                            "marker": f"({token})",
                            "value": token,
                            "kind": self._marker_kind(token),
                            "rank": self._marker_rank(self._marker_kind(token)),
                        }
                        for token in tokens
                    ]
                else:
                    token = tokens[0]
                    kind = self._marker_kind(token)
                    rank = self._marker_rank(kind)

                    while marker_stack and marker_stack[-1]["rank"] >= rank:
                        marker_stack.pop()

                    marker_stack.append(
                        {
                            "marker": f"({token})",
                            "value": token,
                            "kind": kind,
                            "rank": rank,
                        }
                    )

                push_chunk(rest, marker_stack.copy())
                continue

            if current_section_label is None:
                continue

            if current_chunk is not None and self._is_continuation_line(line, current_chunk["text"]):
                current_chunk["text"] += " " + line
                continue

            marker_stack = []
            push_chunk(line, [])

        for idx, chunk in enumerate(chunks, start=1):
            chunk["id"] = idx

        return chunks

    def chunk_markdown_file(
        self,
        markdown_path: str | Path,
        *,
        output_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        markdown_path = Path(markdown_path)
        markdown = markdown_path.read_text(encoding="utf-8")
        chunks = self.chunk_markdown(markdown)

        if output_path is not None:
            self.save_chunks(chunks, output_path)

        return chunks

    @staticmethod
    def save_chunks(chunks: list[dict[str, Any]], output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path

    @staticmethod
    def _normalize_lines(src: str | list[str]) -> list[str]:
        if isinstance(src, str):
            return [line.rstrip() for line in src.splitlines()]
        return [str(line).rstrip("\n\r") for line in src]

    @staticmethod
    def _marker_kind(token: str) -> str:
        if token.isdigit():
            return "number"
        if len(token) == 1 and token.isupper():
            return "upper"
        if token.islower() and re.fullmatch(r"[ivxlcdm]+", token):
            return "roman"
        if len(token) == 1 and token.islower():
            return "lower"
        return "other"

    @staticmethod
    def _marker_rank(kind: str) -> int:
        return {
            "lower": 0,
            "number": 1,
            "roman": 2,
            "upper": 3,
            "other": 4,
        }[kind]

    @staticmethod
    def _is_terminal_chunk(text: str) -> bool:
        return bool(text) and text.rstrip().endswith((".", ";", ":", "?", "!", "]"))

    def _is_continuation_line(self, line: str, previous_text: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False

        if stripped.startswith("§ ") and not self.SECTION_RE.match(stripped):
            return True

        first = stripped[0]
        if first.islower():
            return True
        if first in {",", ";", ":", ")", "]"}:
            return True
        if not self._is_terminal_chunk(previous_text):
            return True

        return False

    @staticmethod
    def _make_chunk(
        *,
        text: str,
        part_label: str | None,
        subpart_label: str | None,
        section_label: str,
        markers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        path = [label for label in [part_label, subpart_label, section_label] if label]
        path.extend(marker["marker"] for marker in markers)

        return {
            "path": path,
            "path_text": " > ".join(path),
            "text": text.strip(),
            "section": section_label,
            "part": part_label,
            "subpart": subpart_label,
            "markers": [marker["marker"] for marker in markers],
        }


if __name__ == "__main__":
    chunker = MarkdownChunker()
    chunks = chunker.chunk_markdown_file("filtered_markdown.md", output_path="chunks.json")
    print(chunks[:3])
