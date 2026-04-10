from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import re

from sqlalchemy import delete, text

from app.config import get_settings
from app.db import SessionLocal
from app.ingest.chunking import MarkdownChunker
from app.models import RetrievalChunk, StructuralContent
from app.schemas.system import IngestionResult, IngestionSummary
from app.schemas.types import StructuralContentTargetEnum
from app.services.embeddings import EmbeddingService
from app.services.text_utils import estimate_token_count


PART_LABEL_RE = re.compile(r"^PART\s+(\d{3})(?:\s*[-\s]?\s*(.*))?$", re.IGNORECASE)
SUBPART_LABEL_RE = re.compile(r"^Subpart\s+([A-Z])(?:\s*-\s*(.*))?$", re.IGNORECASE)
SECTION_LABEL_RE = re.compile(r"^§\s*(\d+\.\d+)\b(?:\s+(.*))?$")
BM25_INDEX_NAME = "retrieval_chunks_search_text_bm25_idx"
BM25_INDEX_SQL = f"""
CREATE INDEX {BM25_INDEX_NAME}
ON retrieval_chunks
USING bm25 (search_text)
WITH (text_config = 'english')
"""

def _normalize_optional(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _load_markdown(markdown_path: str | None) -> tuple[str, str]:
    settings = get_settings()
    source_path = Path(markdown_path or settings.filtered_markdown_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Filtered markdown not found at {source_path}")
    return source_path.read_text(encoding="utf-8"), str(source_path)


def _parse_part_label(label: str | None) -> tuple[str | None, str | None]:
    if not label:
        return None, None
    match = PART_LABEL_RE.match(label.strip())
    if not match:
        return None, None
    return match.group(1), _normalize_optional(match.group(2))


def _parse_subpart_label(label: str | None) -> tuple[str | None, str | None]:
    if not label:
        return None, None
    match = SUBPART_LABEL_RE.match(label.strip())
    if not match:
        return None, None
    return match.group(1).upper(), _normalize_optional(match.group(2))


def _parse_section_label(label: str | None) -> tuple[str | None, str | None]:
    if not label:
        return None, None
    match = SECTION_LABEL_RE.match(label.strip())
    if not match:
        return None, None
    return match.group(1), _normalize_optional(match.group(2))


def _join_outline_lines(lines: list[str]) -> str:
    compact: list[str] = []
    for line in lines:
        normalized = line.rstrip()
        if not normalized:
            if compact and compact[-1]:
                compact.append("")
            continue
        compact.append(normalized)
    while compact and not compact[-1]:
        compact.pop()
    return "\n".join(compact)


def _build_structural_content(
    chunks: list[dict[str, object]],
    *,
    source_path: str,
) -> list[StructuralContent]:
    section_groups: dict[str, dict[str, object]] = {}
    subpart_groups: dict[tuple[str | None, str], dict[str, object]] = {}
    part_groups: dict[str, dict[str, object]] = {}

    for chunk in chunks:
        chunk_text = str(chunk["text"]).strip()
        part_label = _normalize_optional(chunk.get("part"))
        subpart_label = _normalize_optional(chunk.get("subpart"))
        section_label = _normalize_optional(chunk.get("section"))
        if not section_label:
            continue

        part_number, _ = _parse_part_label(part_label)
        subpart_key, _ = _parse_subpart_label(subpart_label)
        section_number, _ = _parse_section_label(section_label)

        section_entry = section_groups.setdefault(
            section_label,
            {
                "path": [label for label in [part_label, subpart_label, section_label] if label],
                "part": part_label,
                "subpart": subpart_label,
                "section": section_label,
                "part_number": part_number,
                "subpart_key": subpart_key,
                "section_number": section_number,
                "section_title": _parse_section_label(section_label)[1],
                "texts": [],
            },
        )
        section_entry["texts"].append(chunk_text)

        if part_label:
            part_entry = part_groups.setdefault(
                part_label,
                {
                    "path": [part_label],
                    "part": part_label,
                    "part_number": part_number,
                    "direct_sections": [],
                    "subparts": {},
                },
            )
            if subpart_label:
                subpart_entry = part_entry["subparts"].setdefault(
                    subpart_label,
                    {
                        "subpart": subpart_label,
                        "subpart_key": subpart_key,
                        "sections": [],
                    },
                )
                if section_label not in {item["section"] for item in subpart_entry["sections"]}:
                    subpart_entry["sections"].append(
                        {
                            "section": section_label,
                            "section_number": section_number,
                            "section_title": _parse_section_label(section_label)[1],
                        }
                    )
            elif section_label not in {item["section"] for item in part_entry["direct_sections"]}:
                part_entry["direct_sections"].append(
                    {
                        "section": section_label,
                        "section_number": section_number,
                        "section_title": _parse_section_label(section_label)[1],
                    }
                )

        if subpart_label:
            subpart_entry = subpart_groups.setdefault(
                (part_label, subpart_label),
                {
                    "path": [label for label in [part_label, subpart_label] if label],
                    "part": part_label,
                    "subpart": subpart_label,
                    "part_number": part_number,
                    "subpart_key": subpart_key,
                    "sections": [],
                },
            )
            if section_label not in {item["section"] for item in subpart_entry["sections"]}:
                subpart_entry["sections"].append(
                    {
                        "section": section_label,
                        "section_number": section_number,
                        "section_title": _parse_section_label(section_label)[1],
                    }
                )

    structural_items: list[StructuralContent] = []

    for section_entry in section_groups.values():
        structural_items.append(
            StructuralContent(
                content_type=StructuralContentTargetEnum.SECTION_TEXT,
                path=list(section_entry["path"]),
                path_text=" > ".join(section_entry["path"]),
                text=f'{section_entry["section"]}\n\n' + "\n\n".join(section_entry["texts"]),
                part=section_entry["part"],
                subpart=section_entry["subpart"],
                section=section_entry["section"],
                part_number=section_entry["part_number"],
                subpart_key=section_entry["subpart_key"],
                section_number=section_entry["section_number"],
                metadata_json={
                    "source_mode": "markdown",
                    "source_path": source_path,
                    "content_type": StructuralContentTargetEnum.SECTION_TEXT.value,
                    "section_title": section_entry["section_title"],
                },
            )
        )

    for subpart_entry in subpart_groups.values():
        lines = [value for value in [subpart_entry["part"], subpart_entry["subpart"], ""] if value is not None]
        lines.extend(str(section["section"]) for section in subpart_entry["sections"])
        structural_items.append(
            StructuralContent(
                content_type=StructuralContentTargetEnum.SUBPART_OUTLINE,
                path=list(subpart_entry["path"]),
                path_text=" > ".join(subpart_entry["path"]),
                text=_join_outline_lines(lines),
                part=subpart_entry["part"],
                subpart=subpart_entry["subpart"],
                section=None,
                part_number=subpart_entry["part_number"],
                subpart_key=subpart_entry["subpart_key"],
                section_number=None,
                metadata_json={
                    "source_mode": "markdown",
                    "source_path": source_path,
                    "content_type": StructuralContentTargetEnum.SUBPART_OUTLINE.value,
                    "sections": subpart_entry["sections"],
                },
            )
        )

    for part_entry in part_groups.values():
        lines = [str(part_entry["part"]), ""]
        if part_entry["direct_sections"]:
            lines.extend(["Sections with no Subpart"])
            lines.extend(str(section["section"]) for section in part_entry["direct_sections"])
            lines.append("")

        for subpart_entry in part_entry["subparts"].values():
            lines.append(str(subpart_entry["subpart"]))
            lines.extend(str(section["section"]) for section in subpart_entry["sections"])
            lines.append("")

        structural_items.append(
            StructuralContent(
                content_type=StructuralContentTargetEnum.PART_OUTLINE,
                path=list(part_entry["path"]),
                path_text=" > ".join(part_entry["path"]),
                text=_join_outline_lines(lines),
                part=part_entry["part"],
                subpart=None,
                section=None,
                part_number=part_entry["part_number"],
                subpart_key=None,
                section_number=None,
                metadata_json={
                    "source_mode": "markdown",
                    "source_path": source_path,
                    "content_type": StructuralContentTargetEnum.PART_OUTLINE.value,
                    "direct_sections": part_entry["direct_sections"],
                    "subparts": list(part_entry["subparts"].values()),
                },
            )
        )

    return structural_items


async def run_ingestion(
    *,
    fake_embeddings: bool = False,
    markdown_path: str | None = None,
) -> IngestionResult:
    markdown, source_path = _load_markdown(markdown_path)

    chunker = MarkdownChunker()
    embedding_service = EmbeddingService(use_fake_embeddings=fake_embeddings)

    chunks = chunker.chunk_markdown(markdown)
    structural_content = _build_structural_content(chunks, source_path=source_path)
    embeddings = await embedding_service.embed_texts([str(chunk["text"]) for chunk in chunks]) if chunks else []

    async with SessionLocal() as session:
        await session.execute(text(f"DROP INDEX IF EXISTS {BM25_INDEX_NAME}"))
        await session.execute(delete(StructuralContent))
        await session.execute(delete(RetrievalChunk))

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk_text = str(chunk["text"]).strip()
            db_chunk = RetrievalChunk(
                id=int(chunk["id"]),
                path=[str(value) for value in chunk.get("path", [])],
                path_text=str(chunk.get("path_text", "")),
                text=chunk_text,
                section=_normalize_optional(chunk.get("section")),
                part=_normalize_optional(chunk.get("part")),
                subpart=_normalize_optional(chunk.get("subpart")),
                markers=[str(value) for value in chunk.get("markers", [])],
                token_count=estimate_token_count(chunk_text),
                metadata_json={
                    "source_mode": "markdown",
                    "source_path": source_path,
                },
                embedding=embedding,
            )
            session.add(db_chunk)

        for structural_item in structural_content:
            session.add(structural_item)

        await session.flush()
        await session.execute(text(BM25_INDEX_SQL))

        summary = IngestionSummary(
            retrieval_chunks=len(chunks),
            lexical_index="pg_textsearch",
            dense_index="pgvector_exact",
            source_mode="markdown",
        )
        await session.commit()

    return IngestionResult(status="completed", summary=summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest HIPAA source data into the database.")
    parser.add_argument(
        "--fake-embeddings",
        action="store_true",
        help="Skip OpenAI and store random embeddings with the configured dimension.",
    )
    parser.add_argument(
        "--markdown-path",
        help="Path to an existing filtered markdown file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(
        run_ingestion(
            fake_embeddings=args.fake_embeddings,
            markdown_path=args.markdown_path,
        )
    )
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
