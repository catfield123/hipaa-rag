"""WebSocket wire formats for ``/rag/query/ws`` (discriminated by ``type``)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.chat import ChatQueryResponse


class WsStatusEvent(BaseModel):
    """Agent progress update streamed before the final ``result`` message.

    Args (fields):
        type (Literal["status"]): Discriminator; always ``status``.
        phase (str): Pipeline phase label (e.g. ``start``, ``retrieve``, ``decide``, ``answer``).
        message (str): Human-readable status line for the UI.
        round_number (int | None): Retrieval round index, serialized as JSON key ``round``.
        tool (str | None): OpenAI tool name when a retrieval tool is active.
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["status"] = "status"
    phase: str
    message: str
    round_number: int | None = Field(default=None, alias="round")
    tool: str | None = None


class WsAnswerDeltaEvent(BaseModel):
    """Incremental final-answer text chunk when streaming is enabled.

    Args (fields):
        type (Literal["answer_delta"]): Discriminator; always ``answer_delta``.
        text (str): Token or substring fragment to append to the running answer.
    """

    type: Literal["answer_delta"] = "answer_delta"
    text: str


class WsErrorEvent(BaseModel):
    """Client-visible error sent before the socket closes.

    Args (fields):
        type (Literal["error"]): Discriminator; always ``error``.
        message (str): Sanitized error text (no stack traces).
    """

    type: Literal["error"] = "error"
    message: str


def ws_result_event_payload(response: ChatQueryResponse) -> dict[str, Any]:
    """Build the final WebSocket message (same logical content as ``POST /rag/query``).

    Args:
        response (ChatQueryResponse): Completed RAG response.

    Returns:
        dict[str, Any]: JSON object with ``type`` set to ``result`` plus response fields.
    """

    return {"type": "result", **response.model_dump(mode="json")}
