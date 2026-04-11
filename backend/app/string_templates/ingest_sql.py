"""SQL fragments for ingestion (index DDL). Identifiers are supplied by callers."""

from __future__ import annotations

BM25_INDEX_DDL = """
CREATE INDEX {index_name}
ON retrieval_chunks
USING bm25 (search_text)
WITH (text_config = 'english')
"""

DROP_INDEX_IF_EXISTS = "DROP INDEX IF EXISTS {index_name}"
