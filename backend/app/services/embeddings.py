from __future__ import annotations

from itertools import islice

from app.config import get_settings
from app.services.openai_client import get_openai_client


class EmbeddingService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = get_openai_client()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        embeddings: list[list[float]] = []
        for batch in self._batched(texts, size=32):
            response = await self.client.embeddings.create(
                model=self.settings.openai_embedding_model,
                input=batch,
            )
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_texts([text]))[0]

    def _batched(self, items: list[str], size: int) -> list[list[str]]:
        iterator = iter(items)
        batches: list[list[str]] = []
        while batch := list(islice(iterator, size)):
            batches.append(batch)
        return batches
