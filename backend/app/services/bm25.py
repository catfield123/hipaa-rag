from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import BM25CorpusStat, BM25Posting, BM25Term, RetrievalChunk
from app.schemas import RetrievalEvidence
from app.services.text_utils import estimate_token_count, tokenize


@dataclass
class BM25BuildResult:
    corpus_stat: dict[str, float | int]
    terms: list[dict[str, float | int | str]]
    postings: list[dict[str, int | str]]
    chunk_lengths: dict[int, int]


class BM25Service:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def build(self, chunks: list[dict[str, object]]) -> BM25BuildResult:
        total_chunks = len(chunks)
        document_frequency: Counter[str] = Counter()
        chunk_terms: dict[int, Counter[str]] = {}
        chunk_lengths: dict[int, int] = {}

        for chunk in chunks:
            chunk_id = int(chunk["chunk_id"])
            terms = tokenize(str(chunk["content"]))
            lexical_length = len(terms)
            if lexical_length == 0:
                continue
            counts = Counter(terms)
            chunk_terms[chunk_id] = counts
            chunk_lengths[chunk_id] = lexical_length
            document_frequency.update(counts.keys())

        average_length = sum(chunk_lengths.values()) / max(len(chunk_lengths), 1)

        terms_payload: list[dict[str, float | int | str]] = []
        postings_payload: list[dict[str, int | str]] = []
        for term, df in document_frequency.items():
            idf = math.log((total_chunks - df + 0.5) / (df + 0.5) + 1)
            terms_payload.append(
                {
                    "term": term,
                    "document_frequency": df,
                    "inverse_document_frequency": idf,
                }
            )

        for chunk_id, counts in chunk_terms.items():
            for term, frequency in counts.items():
                postings_payload.append(
                    {
                        "term": term,
                        "chunk_id": chunk_id,
                        "term_frequency": frequency,
                    }
                )

        return BM25BuildResult(
            corpus_stat={
                "total_chunks": len(chunk_lengths),
                "average_document_length": average_length,
            },
            terms=terms_payload,
            postings=postings_payload,
            chunk_lengths=chunk_lengths,
        )

    async def search(
        self,
        session: AsyncSession,
        query_text: str,
        limit: int,
    ) -> list[RetrievalEvidence]:
        query_terms = tokenize(query_text)
        if not query_terms:
            return []

        corpus_stat = await session.scalar(
            select(BM25CorpusStat).order_by(BM25CorpusStat.id.desc()).limit(1)
        )
        if corpus_stat is None:
            return []

        term_rows = (
            await session.execute(select(BM25Term).where(BM25Term.term.in_(query_terms)))
        ).scalars().all()
        if not term_rows:
            return []

        idf_map = {row.term: row.inverse_document_frequency for row in term_rows}

        postings = (
            await session.execute(
                select(BM25Posting, RetrievalChunk)
                .join(RetrievalChunk, RetrievalChunk.id == BM25Posting.chunk_id)
                .where(BM25Posting.term.in_(idf_map.keys()))
            )
        ).all()

        scores: dict[int, float] = defaultdict(float)
        chunk_map: dict[int, RetrievalChunk] = {}
        for posting, chunk in postings:
            chunk_map[chunk.id] = chunk
            doc_len = max(int(chunk.metadata_json.get("bm25_length", chunk.token_count)), 1)
            idf = idf_map.get(posting.term, 0.0)
            tf = posting.term_frequency
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_len / max(corpus_stat.average_document_length, 1))
            )
            scores[chunk.id] += idf * (numerator / denominator)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [
            RetrievalEvidence(
                chunk_id=chunk_id,
                source_label=chunk_map[chunk_id].source_label,
                page_start=chunk_map[chunk_id].page_start,
                page_end=chunk_map[chunk_id].page_end,
                content=chunk_map[chunk_id].content,
                content_with_context=chunk_map[chunk_id].content_with_context,
                retrieval_mode="bm25_only",
                score=score,
                metadata=chunk_map[chunk_id].metadata_json,
            )
            for chunk_id, score in ranked
        ]


def chunk_payloads_for_bm25(chunks: list[object]) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for chunk in chunks:
        payloads.append(
            {
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "token_count": chunk.get("token_count", estimate_token_count(chunk["content"])),
            }
        )
    return payloads
