"""Deterministic heuristics for structure-aware HIPAA queries."""

from __future__ import annotations

import re

from app.schemas import RetrievalEvidence, StructuralFilters
from app.services.text_utils import tokenize


class StructuralQueryInterpreter:
    """Encapsulate structure-aware query heuristics and evidence ranking rules."""

    def infer_structural_filters(self, user_query: str) -> StructuralFilters | None:
        """Extract structural filters like part, section, subpart, and markers."""

        lowered = user_query.lower()
        part_match = re.search(r"\bpart\s+(\d{3})\b", lowered, flags=re.IGNORECASE)
        section_match = re.search(r"(?:§\s*)?(\d{3}\.\d+)", lowered, flags=re.IGNORECASE)
        subpart_match = re.search(r"\bsubpart\s+([a-z])\b", lowered, flags=re.IGNORECASE)
        marker_path = re.findall(r"\(([A-Za-z0-9ivxlcdmIVXLCDM]+)\)", user_query)

        if not (part_match or section_match or subpart_match or marker_path):
            return None

        return StructuralFilters(
            part_number=part_match.group(1) if part_match else None,
            section_number=section_match.group(1) if section_match else None,
            subpart=subpart_match.group(1).upper() if subpart_match else None,
            marker_path=[marker.lower() for marker in marker_path],
        )

    def is_section_text_request(
        self,
        lowered_query: str,
        inferred_filters: StructuralFilters | None,
    ) -> bool:
        """Return whether the user explicitly asked for full section text."""

        if not inferred_filters or not inferred_filters.section_number:
            return False
        phrases = (
            "full text",
            "entire section",
            "whole section",
            "text of",
            "show me",
            "show ",
            "give me",
            "provide",
            "quote",
        )
        return any(phrase in lowered_query for phrase in phrases)

    def is_direct_part_outline_request(
        self,
        lowered_query: str,
        inferred_filters: StructuralFilters | None,
    ) -> bool:
        """Return whether the user explicitly asked to display a part outline."""

        if not inferred_filters or not inferred_filters.part_number:
            return False
        return self.has_direct_structure_display_signal(lowered_query)

    def is_direct_subpart_outline_request(
        self,
        lowered_query: str,
        inferred_filters: StructuralFilters | None,
    ) -> bool:
        """Return whether the user explicitly asked to display a subpart outline."""

        if not inferred_filters or not inferred_filters.subpart:
            return False
        return self.has_direct_structure_display_signal(lowered_query)

    def is_part_structural_reasoning_request(
        self,
        lowered_query: str,
        inferred_filters: StructuralFilters | None,
    ) -> bool:
        """Return whether a part-level overview should start from the outline."""

        if not inferred_filters or not inferred_filters.part_number:
            return False
        return any(keyword in lowered_query for keyword in self._reasoning_keywords())

    def is_subpart_structural_reasoning_request(
        self,
        lowered_query: str,
        inferred_filters: StructuralFilters | None,
    ) -> bool:
        """Return whether a subpart-level overview should start from the outline."""

        if not inferred_filters or not inferred_filters.subpart:
            return False
        return any(keyword in lowered_query for keyword in self._reasoning_keywords())

    def should_return_raw_structure(self, question: str) -> bool:
        """Return whether the final answer should be raw structural content."""

        lowered_query = question.lower()
        inferred_filters = self.infer_structural_filters(question)
        return (
            self.is_section_text_request(lowered_query, inferred_filters)
            or self.is_direct_part_outline_request(lowered_query, inferred_filters)
            or self.is_direct_subpart_outline_request(lowered_query, inferred_filters)
        )

    def has_direct_structure_display_signal(self, lowered_query: str) -> bool:
        """Return whether the prompt asks to show or list structure directly."""

        direct_phrases = (
            "list",
            "show",
            "display",
            "outline",
            "contents",
            "table of contents",
            "section headings",
            "all sections",
            "which sections",
        )
        return any(phrase in lowered_query for phrase in direct_phrases)

    def outline_sections(self, evidence: RetrievalEvidence) -> list[dict[str, object]]:
        """Extract outline sections from structural evidence metadata."""

        content_type = evidence.metadata.get("content_type")
        if content_type == "subpart_outline":
            sections = evidence.metadata.get("sections")
            return [item for item in sections if isinstance(item, dict)] if isinstance(sections, list) else []

        if content_type == "part_outline":
            sections: list[dict[str, object]] = []
            direct_sections = evidence.metadata.get("direct_sections")
            if isinstance(direct_sections, list):
                sections.extend(item for item in direct_sections if isinstance(item, dict))
            subparts = evidence.metadata.get("subparts")
            if isinstance(subparts, list):
                for subpart in subparts:
                    if not isinstance(subpart, dict):
                        continue
                    subpart_sections = subpart.get("sections")
                    if isinstance(subpart_sections, list):
                        sections.extend(item for item in subpart_sections if isinstance(item, dict))
            return sections

        return []

    def best_outline_sections(
        self,
        question: str,
        evidence: RetrievalEvidence,
    ) -> list[dict[str, object]]:
        """Rank outline sections by overlap with the current question."""

        candidates = self.outline_sections(evidence)
        if not candidates:
            return []

        question_tokens = self._question_tokens(question)
        ranked: list[tuple[int, dict[str, object]]] = []
        lowered_question = question.lower()
        for section in candidates:
            section_title = str(section.get("section_title") or "")
            title_tokens = set(tokenize(section_title))
            score = len(question_tokens & title_tokens)
            if "purpose" in lowered_question and "purpose" in section_title.lower():
                score += 3
            if "definition" in lowered_question and "definition" in section_title.lower():
                score += 3
            if "scope" in lowered_question and "scope" in section_title.lower():
                score += 3
            if score > 0:
                ranked.append((score, section))

        ranked.sort(
            key=lambda item: (
                -item[0],
                str(item[1].get("section_number") or ""),
            )
        )
        return [section for _, section in ranked]

    def best_referenced_sections(
        self,
        question: str,
        evidence: list[RetrievalEvidence],
    ) -> list[dict[str, object]]:
        """Rank referenced sections from chunk evidence for follow-up lookups."""

        question_tokens = self._question_tokens(question)
        lowered_question = question.lower()
        grouped: dict[str, dict[str, object]] = {}
        for item in evidence:
            section_number, section_title = self.parse_section_reference(item.section)
            if not section_number:
                continue

            entry = grouped.setdefault(
                section_number,
                {
                    "section_number": section_number,
                    "section_title": section_title,
                    "score": 0.0,
                    "count": 0,
                },
            )
            entry["count"] = int(entry["count"]) + 1
            entry["score"] = float(entry["score"]) + float(item.score)

            title_tokens = set(tokenize(section_title))
            entry["score"] = float(entry["score"]) + float(len(question_tokens & title_tokens))

            item_text = str(item.text or "").lower()
            if "family" in lowered_question and "family" in item_text:
                entry["score"] = float(entry["score"]) + 2.0
            if "law enforcement" in lowered_question and "law enforcement" in item_text:
                entry["score"] = float(entry["score"]) + 2.0
            if "purpose" in lowered_question and "purpose" in section_title.lower():
                entry["score"] = float(entry["score"]) + 3.0
            if "definition" in lowered_question and "definition" in section_title.lower():
                entry["score"] = float(entry["score"]) + 3.0
            if "scope" in lowered_question and "applicability" in section_title.lower():
                entry["score"] = float(entry["score"]) + 2.0

        return sorted(
            grouped.values(),
            key=lambda item: (
                -float(item["score"]),
                -int(item["count"]),
                str(item["section_number"]),
            ),
        )

    def parse_section_reference(self, section_label: str | None) -> tuple[str | None, str]:
        """Split a rendered section label into number and title."""

        if not section_label:
            return None, ""
        match = re.match(r"^§\s*(\d+\.\d+)\b\s*(.*)$", section_label.strip())
        if not match:
            return None, section_label.strip()
        return match.group(1), match.group(2).strip()

    def _question_tokens(self, question: str) -> set[str]:
        tokens = set(tokenize(question))
        if tokens:
            return tokens
        return {token.lower() for token in re.findall(r"[a-z][a-z0-9']*", question.lower())}

    def _reasoning_keywords(self) -> tuple[str, ...]:
        return ("purpose", "overview", "cover", "covers", "about")
