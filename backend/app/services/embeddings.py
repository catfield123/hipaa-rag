"""Embedding helpers for query and document vectors."""

from __future__ import annotations

import math
import random
from itertools import islice

from openai import AsyncOpenAI

from app.config import get_settings
from app.core.exceptions import ConfigurationError
from app.services.openai_client import get_openai_client


class EmbeddingService:
    """Create embeddings through OpenAI or a deterministic fake generator."""

    def __init__(self, *, use_fake_embeddings: bool = False) -> None:
        """Initialize the embedding service.

        Args:
            use_fake_embeddings (bool): If True, use deterministic local vectors instead of OpenAI.

        Returns:
            None

        Raises:
            None
        """

        self.settings = get_settings()
        self.use_fake_embeddings = use_fake_embeddings
        self._client: AsyncOpenAI | None = None
        self._random = random.Random()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings.

        Args:
            texts (list[str]): Non-empty batch segments to embed.

        Returns:
            list[list[float]]: One embedding vector per input string.

        Raises:
            ConfigurationError: If ``OPENAI_API_KEY`` is unset and fake embeddings are disabled.
        """

        if self.use_fake_embeddings:
            return [self._fake_embedding() for _ in texts]

        if not self.settings.openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY is not configured.")

        embeddings: list[list[float]] = []
        for batch in self._batched(texts, size=32):
            kwargs: dict = {
                "model": self.settings.openai_embedding_model,
                "input": batch,
            }
            if self.settings.openai_embedding_model.startswith("text-embedding-3"):
                kwargs["dimensions"] = self.settings.embedding_dimension
            response = await self._get_client().embeddings.create(**kwargs)
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text (str): Query text.

        Returns:
            list[float]: Embedding vector.

        Raises:
            ConfigurationError: If ``OPENAI_API_KEY`` is unset and fake embeddings are disabled.
        """

        return (await self.embed_texts([text]))[0]

    def _batched(self, items: list[str], size: int) -> list[list[str]]:
        """Split a list into contiguous batches of at most ``size`` items.

        Args:
            items (list[str]): All strings to embed in order.
            size (int): Batch size (e.g. 32 for OpenAI).

        Returns:
            list[list[str]]: Non-empty batches covering ``items``.

        Raises:
            None
        """

        iterator = iter(items)
        batches: list[list[str]] = []
        while batch := list(islice(iterator, size)):
            batches.append(batch)
        return batches

    def _get_client(self) -> AsyncOpenAI:
        """Lazily construct and cache the process-wide async OpenAI client.

        Args:
            None

        Returns:
            AsyncOpenAI: Client from :func:`app.services.openai_client.get_openai_client`.

        Raises:
            None
        """

        if self._client is None:
            self._client = get_openai_client()
        return self._client

    def _fake_embedding(self) -> list[float]:
        """Generate a random unit vector of length ``settings.embedding_dimension``.

        Args:
            None

        Returns:
            list[float]: L2-normalized vector suitable for pgvector storage.

        Raises:
            None
        """

        values = [
            self._random.uniform(-1.0, 1.0)
            for _ in range(self.settings.embedding_dimension)
        ]
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            values[0] = 1.0
            norm = 1.0
        return [value / norm for value in values]
