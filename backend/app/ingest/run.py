from __future__ import annotations

import asyncio
import json
from pathlib import Path

from sqlalchemy import delete

from app.config import get_settings
from app.db import SessionLocal
from app.models import BM25CorpusStat, BM25Posting, BM25Term, DocumentNode, RetrievalChunk
from app.schemas import IngestionResult, IngestionSummary
from app.services.bm25 import BM25Service, chunk_payloads_for_bm25
from app.services.chunking import RetrievalChunker
from app.services.embeddings import EmbeddingService
from app.services.pdf_parser import HIPAAPdfParser


async def run_ingestion() -> IngestionResult:
    settings = get_settings()
    pdf_path = Path(settings.hipaa_pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"HIPAA PDF not found at {pdf_path}")

    parser = HIPAAPdfParser()
    chunker = RetrievalChunker()
    embedding_service = EmbeddingService()
    bm25_service = BM25Service()

    parsed_document = parser.parse(str(pdf_path))
    chunks = chunker.build_chunks(parsed_document)
    embeddings = await embedding_service.embed_texts([chunk.content_with_context for chunk in chunks])

    async with SessionLocal() as session:
        await session.execute(delete(BM25Posting))
        await session.execute(delete(BM25Term))
        await session.execute(delete(BM25CorpusStat))
        await session.execute(delete(RetrievalChunk))
        await session.execute(delete(DocumentNode))
        await session.flush()

        node_id_map: dict[str, int] = {}
        for node in parsed_document.nodes:
            db_node = DocumentNode(
                parent_id=node_id_map.get(node.parent_key) if node.parent_key else None,
                node_type=node.node_type,
                part_number=node.part_number,
                subpart=node.subpart,
                section_number=node.section_number,
                marker=node.marker,
                source_label=node.source_label,
                heading=node.heading,
                raw_text=node.raw_text,
                page_start=node.page_start,
                page_end=node.page_end,
            )
            session.add(db_node)
            await session.flush()
            node_id_map[node.key] = db_node.id

        chunk_payloads: list[dict[str, object]] = []
        persisted_chunks: list[RetrievalChunk] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            db_chunk = RetrievalChunk(
                start_node_id=node_id_map[chunk.start_node_key],
                end_node_id=node_id_map[chunk.end_node_key],
                chunk_index=chunk.chunk_index,
                source_label=chunk.source_label,
                content=chunk.content,
                content_with_context=chunk.content_with_context,
                token_count=chunk.token_count,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                metadata_json={
                    **chunk.metadata,
                    "start_node_id": node_id_map[chunk.start_node_key],
                    "end_node_id": node_id_map[chunk.end_node_key],
                    "quote_node_id": node_id_map[chunk.quote_node_key],
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                },
                embedding=embedding,
            )
            session.add(db_chunk)
            await session.flush()
            persisted_chunks.append(db_chunk)
            chunk_payloads.append(
                {
                    "chunk_id": db_chunk.id,
                    "content": db_chunk.content,
                    "token_count": db_chunk.token_count,
                }
            )

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
            document_nodes=len(parsed_document.nodes),
            retrieval_chunks=len(chunks),
            bm25_terms=len(bm25_build.terms),
        )
        await session.commit()

    return IngestionResult(status="completed", summary=summary)


def main() -> None:
    result = asyncio.run(run_ingestion())
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
