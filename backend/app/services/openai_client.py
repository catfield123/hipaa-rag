"""Shared OpenAI client factory."""

from functools import lru_cache

from openai import AsyncOpenAI

from app.config import get_settings


@lru_cache
def get_openai_client() -> AsyncOpenAI:
    """Return a cached async OpenAI client configured from settings."""

    settings = get_settings()
    return AsyncOpenAI(api_key=settings.openai_api_key)
