"""Chat API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.chat import ChatQueryHandler
from app.db import get_db_session
from app.schemas.chat import ChatQueryRequest, ChatQueryResponse

router = APIRouter(prefix="/chat", tags=["chat"])
DbSessionDep = Annotated[AsyncSession, Depends(get_db_session)]


def get_chat_query_handler() -> ChatQueryHandler:
    """Provide the chat query application service."""

    return ChatQueryHandler()


ChatQueryHandlerDep = Annotated[ChatQueryHandler, Depends(get_chat_query_handler)]


@router.post("/query")
async def query_chat(
    payload: ChatQueryRequest,
    session: DbSessionDep,
    handler: ChatQueryHandlerDep,
) -> ChatQueryResponse:
    """Run the multi-round retrieval and answer synthesis workflow."""

    return await handler.handle_query(
        question=payload.question,
        include_debug=payload.include_debug,
        session=session,
    )
