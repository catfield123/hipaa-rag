"""WebSocket event ``type`` values for the RAG streaming API (frontend mirror of backend)."""

from __future__ import annotations

from enum import StrEnum


class RagWsEventType(StrEnum):
    """Values for the JSON ``type`` field on RAG WebSocket messages.

    Mirrors :class:`app.schemas.types.RagWsEventType` in the backend.
    """

    STATUS = "status"
    ANSWER_DELTA = "answer_delta"
    ERROR = "error"
    RESULT = "result"
