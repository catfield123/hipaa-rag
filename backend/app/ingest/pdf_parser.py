"""PDF-to-markdown conversion helpers used by the ingestion pipeline.

Requires the optional ``docling`` package at runtime for :class:`PdfToMarkdownConverter`
unless a compatible converter instance is injected via the constructor.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.string_templates.pdf_inline import INLINE_PAREN_WRAPPED

HIPAA_PART_160_PREFIX = """PART 160 GENERAL ADMINISTRATIVE REQUIREMENTS

Subpart A-General Provisions

§ 160.101   Statutory basis and purpose.

The requirements of this subchapter implement sections 1171-1180 of the Social Security Act (the Act), sections 262 and 264 of Public Law 104-191, section 105 of Public Law110-233, sections 13400-13424of Public Law 111-5, andsection 1104 of Public Law 111-148.[78 FR 5687, Jan. 25, 2013]
"""


class PdfToMarkdownConverter:
    """Convert a HIPAA source PDF into normalized markdown suitable for :class:`MarkdownChunker`."""

    def __init__(self, converter: Any | None = None) -> None:
        """Create a converter, optionally reusing an existing Docling ``DocumentConverter``.

        Args:
            converter (Any | None): Pre-built Docling converter, or ``None`` to build one lazily.

        Returns:
            None

        Raises:
            RuntimeError: If ``converter`` is ``None`` and the ``docling`` package is not installed.
        """

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
        """Convert a PDF to markdown and optionally normalize it for deterministic chunking.

        Args:
            pdf_path (str | Path): Path to the source ``.pdf`` file.
            output_path (str | Path | None): If set, writes the final markdown to this path (UTF-8).
            normalize_for_chunking (bool): If ``True``, runs :meth:`normalize_markdown` with ``start_line`` / prepend.
            start_line (str): First line (after strip) used to locate the start of regulated text in the export.
            prepend_text (str | None): Extra text inserted before normalized content (after the fixed Part 160 prefix).

        Returns:
            str: Markdown string (and optionally persisted when ``output_path`` is set).

        Raises:
            RuntimeError: When ``docling`` is missing and no converter was injected (from :meth:`_build_converter`).
            OSError: If the PDF cannot be read or ``output_path`` cannot be written.
            Exception: Implementations may propagate conversion errors from Docling.
        """

        pdf_path = Path(pdf_path)
        result = self.converter.convert(str(pdf_path))
        markdown = result.document.export_to_markdown()

        if normalize_for_chunking:
            markdown = self.normalize_markdown(
                markdown,
                start_line=start_line,
                prepend_text=prepend_text,
            )
        else:
            built = self._build_prepend_text(prepend_text=prepend_text)
            if built:
                markdown = built + markdown

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
        """Normalize Docling-exported markdown so :class:`MarkdownChunker` can parse it deterministically.

        Truncates content before ``start_line``, drops table rows, flattens heading/bullet prefixes, and
        tightens parenthetical citation markers.

        Args:
            markdown (str): Raw markdown from ``export_to_markdown``.
            start_line (str): Line prefix marking where normalized regulation text begins.
            prepend_text (str | None): Optional extra prefix (typically from :meth:`_build_prepend_text`); may be ``None``.

        Returns:
            str: Normalized single string (possibly empty if ``start_line`` is not found).

        Raises:
            None
        """

        md_lines = markdown.splitlines()

        try:
            start_index = next(i for i, line in enumerate(md_lines) if line.strip().startswith(start_line))
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
                        lambda match: INLINE_PAREN_WRAPPED.format(
                            inner=match.group(1).replace(" ", ""),
                        ),
                        line,
                        count=1,
                    )

                filtered_lines.append(line.strip())

            normalized = "\n".join(filtered_lines)

        prepend_value = PdfToMarkdownConverter._build_prepend_text(prepend_text=prepend_text)

        if prepend_value:
            normalized = prepend_value + normalized

        return normalized

    @staticmethod
    def save_markdown(markdown: str, output_path: str | Path) -> Path:
        """Write markdown to disk as UTF-8 text.

        Args:
            markdown (str): Full markdown document.
            output_path (str | Path): Destination file path.

        Returns:
            Path: Resolved ``output_path``.

        Raises:
            OSError: If the file cannot be written.
        """

        output_path = Path(output_path)
        output_path.write_text(markdown, encoding="utf-8")
        return output_path

    @staticmethod
    def _build_prepend_text(*, prepend_text: str | None) -> str:
        """Build the fixed Part 160 preamble plus optional caller prepend text.

        Args:
            prepend_text (str | None): Optional fragment inserted after the fixed preamble.

        Returns:
            str: Non-empty prefix string (always includes :data:`HIPAA_PART_160_PREFIX`).

        Raises:
            None
        """

        parts: list[str] = [HIPAA_PART_160_PREFIX]
        if prepend_text:
            parts.append(prepend_text)
        return "".join(parts)

    @staticmethod
    def _build_converter() -> Any:
        """Instantiate Docling's ``DocumentConverter``.

        Args:
            None

        Returns:
            Any: A ``DocumentConverter`` instance.

        Raises:
            RuntimeError: If ``docling`` is not installed.
        """

        try:
            from docling.document_converter import DocumentConverter
        except ModuleNotFoundError as exc:
            raise RuntimeError("PdfToMarkdownConverter requires the optional 'docling' package.") from exc
        return DocumentConverter()


if __name__ == "__main__":
    converter = PdfToMarkdownConverter()
    md = converter.convert_pdf_to_markdown(
        "hipaa-combined.pdf",
        output_path="filtered_markdown.md",
    )
    print(md[:1000])
