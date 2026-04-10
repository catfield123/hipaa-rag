from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from sqlalchemy import delete

from app.config import get_settings
from app.db import SessionLocal
from app.models import BM25CorpusStat, BM25Posting, BM25Term, RetrievalChunk
from app.schemas import IngestionResult, IngestionSummary
from app.services.bm25 import BM25Service, chunk_payloads_for_bm25
from app.services.chunking import MarkdownChunker
from app.services.embeddings import EmbeddingService
from app.services.text_utils import estimate_token_count

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


async def run_ingestion(
    *,
    fake_embeddings: bool = False,
    markdown_path: str | None = None,
) -> IngestionResult:
    markdown, source_path = _load_markdown(markdown_path)

    chunker = MarkdownChunker()
    embedding_service = EmbeddingService(use_fake_embeddings=fake_embeddings)
    bm25_service = BM25Service()

    chunks = chunker.chunk_markdown(markdown)
    embeddings = await embedding_service.embed_texts([str(chunk["text"]) for chunk in chunks]) if chunks else []

    async with SessionLocal() as session:
        await session.execute(delete(BM25Posting))
        await session.execute(delete(BM25CorpusStat))
        await session.execute(delete(BM25Term))
        await session.execute(delete(RetrievalChunk))

        chunk_payloads: list[dict[str, object]] = []
        persisted_chunks: list[RetrievalChunk] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            text = str(chunk["text"]).strip()
            db_chunk = RetrievalChunk(
                id=int(chunk["id"]),
                path=[str(value) for value in chunk.get("path", [])],
                path_text=str(chunk.get("path_text", "")),
                text=text,
                section=_normalize_optional(chunk.get("section")),
                part=_normalize_optional(chunk.get("part")),
                subpart=_normalize_optional(chunk.get("subpart")),
                markers=[str(value) for value in chunk.get("markers", [])],
                token_count=estimate_token_count(text),
                metadata_json={
                    "source_mode": "markdown",
                    "source_path": source_path,
                },
                embedding=embedding,
            )
            session.add(db_chunk)
            persisted_chunks.append(db_chunk)
            chunk_payloads.append(
                {
                    "chunk_id": db_chunk.id,
                    "content": db_chunk.text,
                    "token_count": db_chunk.token_count,
                }
            )

        await session.flush()
        bm25_build = bm25_service.build(chunk_payloads_for_bm25(chunk_payloads))
        for db_chunk in persisted_chunks:
            bm25_length = bm25_build.chunk_lengths.get(db_chunk.id)
            if bm25_length is None:
                continue
            db_chunk.metadata_json = {
                **db_chunk.metadata_json,
                "bm25_length": bm25_length,
            }
        session.add(
            BM25CorpusStat(
                total_chunks=int(bm25_build.corpus_stat["total_chunks"]),
                average_document_length=float(bm25_build.corpus_stat["average_document_length"]),
            )
        )
        session.add_all(
            BM25Term(
                term=item["term"],
                document_frequency=int(item["document_frequency"]),
                inverse_document_frequency=float(item["inverse_document_frequency"]),
            )
            for item in bm25_build.terms
        )
        await session.flush()
        session.add_all(
            BM25Posting(
                term=str(item["term"]),
                chunk_id=int(item["chunk_id"]),
                term_frequency=int(item["term_frequency"]),
            )
            for item in bm25_build.postings
        )

        summary = IngestionSummary(
            retrieval_chunks=len(chunks),
            bm25_terms=len(bm25_build.terms),
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
