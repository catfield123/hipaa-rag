"""WebSocket wire formats for ``/rag/query/ws`` (discriminated by ``type``)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.chat import ChatQueryResponse
from app.schemas.types import AgentPipelinePhaseEnum, RagWsEventType


class WsStatusEvent(BaseModel):
    """Agent progress update streamed before the final ``result`` message.

    Args (fields):
        type (RagWsEventType): Discriminator; always :attr:`~app.schemas.types.RagWsEventType.STATUS`.
        phase (AgentPipelinePhaseEnum): Pipeline step.
        message (str): Human-readable status line for the UI.
        round_number (int | None): Retrieval round index, serialized as JSON key ``round``.
        tool (str | None): OpenAI tool name when a retrieval tool is active.
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal[RagWsEventType.STATUS] = Field(
        default=RagWsEventType.STATUS,
        description="Discriminator; always `status`.",
    )
    phase: AgentPipelinePhaseEnum = Field(description="Pipeline step for this status line.")
    message: str = Field(description="Human-readable status line for the UI.")
    round_number: int | None = Field(
        default=None,
        alias="round",
        description="Retrieval round index when applicable.",
    )
    tool: str | None = Field(
        default=None,
        description="OpenAI tool name when a retrieval tool is active.",
    )


class WsAnswerDeltaEvent(BaseModel):
    """Incremental final-answer text chunk when streaming is enabled.

    Args (fields):
        type (RagWsEventType): Discriminator; always :attr:`~app.schemas.types.RagWsEventType.ANSWER_DELTA`.
        text (str): Token or substring fragment to append to the running answer.
    """

    type: Literal[RagWsEventType.ANSWER_DELTA] = Field(
        default=RagWsEventType.ANSWER_DELTA,
        description="Discriminator; always `answer_delta`.",
    )
    text: str = Field(description="Token or substring fragment to append to the running answer.")


class WsErrorEvent(BaseModel):
    """Client-visible error sent before the socket closes.

    Args (fields):
        type (RagWsEventType): Discriminator; always :attr:`~app.schemas.types.RagWsEventType.ERROR`.
        message (str): Sanitized error text (no stack traces).
    """

    type: Literal[RagWsEventType.ERROR] = Field(
        default=RagWsEventType.ERROR,
        description="Discriminator; always `error`.",
    )
    message: str = Field(description="Sanitized error text (no stack traces).")


def ws_result_event_payload(response: ChatQueryResponse) -> dict[str, Any]:
    """Build the final WebSocket message (same logical content as ``POST /rag/query``).

    Args:
        response (ChatQueryResponse): Completed RAG response.

    Returns:
        dict[str, Any]: JSON object with ``type`` set to ``result`` plus response fields.
    """

    return {"type": RagWsEventType.RESULT, **response.model_dump(mode="json")}
