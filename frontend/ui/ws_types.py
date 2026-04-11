from enum import StrEnum


class RagWsEventType(StrEnum):
    """Mirror of ``app.schemas.types.RagWsEventType`` (WebSocket ``type`` field)."""

    STATUS = "status"
    ANSWER_DELTA = "answer_delta"
    ERROR = "error"
    RESULT = "result"
