"""PDF-to-markdown conversion helpers used by the ingestion pipeline."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter

HIPAA_PART_160_PREFIX = """PART 160 GENERAL ADMINISTRATIVE REQUIREMENTS

Subpart A-General Provisions

§ 160.101   Statutory basis and purpose.

The requirements of this subchapter implement sections 1171-1180 of the Social Security Act (the Act), sections 262 and 264 of Public Law 104-191, section 105 of Public Law110-233, sections 13400-13424of Public Law 111-5, andsection 1104 of Public Law 111-148.[78 FR 5687, Jan. 25, 2013]
"""


class PdfToMarkdownConverter:
    """Convert the source HIPAA PDF into normalized markdown for ingestion."""

    def __init__(self, converter: Any | None = None) -> None:
        self.converter = converter or self._build_converter()

    def convert_pdf_to_markdown(
        self,
        pdf_path: str | Path,
        *,
        output_path: str | Path | None = None,
        normalize_for_chunking: bool = True,
        start_line: str = "## § 160.102   Applicability.",
        prepend_text: str | None = None,
    ) -> str:
        """Convert a PDF to markdown and optionally normalize it for chunking."""

        pdf_path = Path(pdf_path)
        result = self.converter.convert(str(pdf_path))
        markdown = result.document.export_to_markdown()

        prepend_value = self._build_prepend_text(
            prepend_text=prepend_text,
        )

        if normalize_for_chunking:
            markdown = self.normalize_markdown(
                markdown,
                start_line=start_line,
                prepend_text=prepend_value,
            )
        elif prepend_value:
            markdown = prepend_value + markdown

        if output_path is not None:
            output_path = Path(output_path)
            output_path.write_text(markdown, encoding="utf-8")

        return markdown

    @staticmethod
    def normalize_markdown(
        markdown: str,
        *,
        start_line: str = "## § 160.102   Applicability.",
        prepend_text: str | None = None,
    ) -> str:
        """Normalize markdown so it is easier to chunk deterministically."""

        md_lines = markdown.splitlines()

        try:
            start_index = next(
                i for i, line in enumerate(md_lines) if line.strip().startswith(start_line)
            )
        except StopIteration:
            normalized = ""
        else:
            filtered_lines: list[str] = []

            for line in md_lines[start_index:]:
                stripped = line.strip()

                if stripped.startswith("|") and stripped.endswith("|"):
                    continue

                if line.startswith("#"):
                    line = line.lstrip("# ")

                if line.startswith("- "):
                    line = line[2:]

                if line.startswith("("):
                    line = re.sub(
                        r"^\((.*?)\)",
                        lambda match: f"({match.group(1).replace(' ', '')})",
                        line,
                        count=1,
                    )

                filtered_lines.append(line.strip())

            normalized = "\n".join(filtered_lines)

        prepend_value = PdfToMarkdownConverter._build_prepend_text(
            prepend_text=prepend_text,
        )

        if prepend_value:
            normalized = prepend_value + normalized

        return normalized

    @staticmethod
    def save_markdown(markdown: str, output_path: str | Path) -> Path:
        """Write markdown to disk."""

        output_path = Path(output_path)
        output_path.write_text(markdown, encoding="utf-8")
        return output_path

    @staticmethod
    def _build_prepend_text(
        *,
        prepend_text: str | None,
    ) -> str | None:
        parts: list[str] = [HIPAA_PART_160_PREFIX]

        if prepend_text:
            parts.append(prepend_text)

        if not parts:
            return None

        return "".join(parts)

    @staticmethod
    def _build_converter() -> Any:
        try:
            from docling.document_converter import DocumentConverter
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PdfToMarkdownConverter requires the optional 'docling' package."
            ) from exc
        return DocumentConverter()


if __name__ == "__main__":
    converter = PdfToMarkdownConverter()
    md = converter.convert_pdf_to_markdown(
        "hipaa-combined.pdf",
        output_path="filtered_markdown.md",
    )
    print(md[:1000])
