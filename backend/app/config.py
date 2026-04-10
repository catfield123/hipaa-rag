from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "HIPAA RAG Backend"
    api_root_path: str = "/api"
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
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
        return self.database_url.replace("+asyncpg", "+psycopg")

    @property
    def psycopg_connect_url(self) -> str:
        return (
            self.database_url
            .replace("+asyncpg", "")
            .replace("+psycopg", "")
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
