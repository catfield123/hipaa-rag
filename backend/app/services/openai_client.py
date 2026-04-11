"""Shared OpenAI client factory."""

from functools import lru_cache

from app.config import get_settings
from openai import AsyncOpenAI


@lru_cache
def get_openai_client() -> AsyncOpenAI:
    """Return a cached async OpenAI client configured from settings.

    Args:
        None

    Returns:
        AsyncOpenAI: Client using ``openai_api_key`` from :class:`Settings` (may be empty).

    Raises:
        None
    """

    settings = get_settings()
    return AsyncOpenAI(api_key=settings.openai_api_key)
