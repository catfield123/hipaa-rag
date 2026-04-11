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

    app_name: str = "HIPAA RAG Backend"
    api_root_path: str = "/api"
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-large"
    embedding_dimension: int = 3072
    query_rewrite_limit: int = 5
    retrieval_limit: int = 12
    dense_limit: int = 24
    bm25_limit: int = 24
    hybrid_rrf_k: int = 60
    agent_max_rounds: int = 5
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@db:5432/hipaa_rag"
    )
    filtered_markdown_path: str = "/data/filtered_markdown.md"

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

        return (
            self.database_url
            .replace("+asyncpg", "")
            .replace("+psycopg", "")
        )


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
