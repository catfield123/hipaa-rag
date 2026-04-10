"""Embedding helpers for query and document vectors."""

from __future__ import annotations

import math
import random
from itertools import islice

from openai import AsyncOpenAI

from app.config import get_settings
from app.services.openai_client import get_openai_client


class EmbeddingService:
    """Create embeddings through OpenAI or a deterministic fake generator."""

    def __init__(self, *, use_fake_embeddings: bool = False) -> None:
        self.settings = get_settings()
        self.use_fake_embeddings = use_fake_embeddings
        self._client: AsyncOpenAI | None = None
        self._random = random.Random()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings."""

        if self.use_fake_embeddings:
            return [self._fake_embedding() for _ in texts]

        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        embeddings: list[list[float]] = []
        for batch in self._batched(texts, size=32):
            response = await self._get_client().embeddings.create(
                model=self.settings.openai_embedding_model,
                input=batch,
            )
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""

        return (await self.embed_texts([text]))[0]

    def _batched(self, items: list[str], size: int) -> list[list[str]]:
        """Split items into fixed-size batches."""

        iterator = iter(items)
        batches: list[list[str]] = []
        while batch := list(islice(iterator, size)):
            batches.append(batch)
        return batches

    def _get_client(self) -> AsyncOpenAI:
        """Return a cached async OpenAI client."""

        if self._client is None:
            self._client = get_openai_client()
        return self._client

    def _fake_embedding(self) -> list[float]:
        """Generate a normalized fake embedding for local non-OpenAI flows."""

        values = [
            self._random.uniform(-1.0, 1.0)
            for _ in range(self.settings.embedding_dimension)
        ]
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            values[0] = 1.0
            norm = 1.0
        return [value / norm for value in values]
