"""Application settings and configuration helpers."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables and optional ``.env``.

    Args (fields):
        See attributes below; URLs and model names drive OpenAI, SQLAlchemy, and ingestion defaults.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(
        default="HIPAA RAG Backend",
        description="Short name for logs and OpenAPI metadata.",
    )
    api_root_path: str = Field(
        default="/api",
        description="URL prefix for HTTP routes (e.g. reverse-proxy mount path).",
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key; empty disables live LLM calls if your deployment allows.",
    )
    openai_chat_model: str = Field(
        default="gpt-4.1-mini",
        description="Chat/completions model id for answering and agent steps.",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-large",
        description="Embedding model id for dense retrieval and ingestion.",
    )
    embedding_dimension: int = Field(
        default=3072,
        description="Vector size for stored embeddings (must match the embedding model).",
    )
    query_rewrite_limit: int = Field(
        default=5,
        description="Maximum query rewrite variants to consider per request.",
    )
    retrieval_limit: int = Field(
        default=12,
        description="Default cap on chunks merged into the final answer context.",
    )
    dense_limit: int = Field(
        default=24,
        description="Max dense (vector) hits before fusion or reranking.",
    )
    bm25_limit: int = Field(
        default=24,
        description="Max BM25 / lexical hits before fusion or reranking.",
    )
    hybrid_rrf_k: int = Field(
        default=60,
        description="Reciprocal rank fusion `k` for hybrid dense + BM25 lists.",
    )
    agent_max_rounds: int = Field(
        default=5,
        description="Upper bound on retrieval/answer rounds for the agent loop.",
    )
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@db:5432/hipaa_rag",
        description="Async SQLAlchemy database URL (asyncpg driver).",
    )
    filtered_markdown_path: str = Field(
        default="/data/filtered_markdown.md",
        description="Filesystem path to the filtered markdown used for ingestion.",
    )

    @property
    def alembic_database_url(self) -> str:
        """Synchronous SQLAlchemy URL for Alembic (``psycopg`` driver).

        Args:
            None

        Returns:
            str: ``database_url`` with ``+asyncpg`` replaced by ``+psycopg``.

        Raises:
            None
        """

        return self.database_url.replace("+asyncpg", "+psycopg")

    @property
    def psycopg_connect_url(self) -> str:
        """Plain connection string for ``psycopg`` (e.g. :mod:`wait_for_db`).

        Args:
            None

        Returns:
            str: URL without SQLAlchemy async driver prefix.

        Raises:
            None
        """

        return self.database_url.replace("+asyncpg", "").replace("+psycopg", "")


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings object for the current process.

    Args:
        None

    Returns:
        Settings: Loaded from environment and optional ``.env`` file.

    Raises:
        None
    """

    return Settings()
