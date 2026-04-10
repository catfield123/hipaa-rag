from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db_session
from app.schemas import FetchSpanRequest, HealthResponse, NodeResponse, QuoteSpan
from app.services.node_fetcher import NodeFetcher


router = APIRouter(tags=["admin"])
DbSessionDep = Annotated[AsyncSession, Depends(get_db_session)]


@router.get("/health")
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/admin/nodes/{node_id}")
async def get_node(node_id: int, session: DbSessionDep) -> NodeResponse:
    return await NodeFetcher().get_node(session=session, node_id=node_id)


@router.post("/admin/fetch-span")
async def fetch_span(payload: FetchSpanRequest, session: DbSessionDep) -> QuoteSpan:
    return await NodeFetcher().get_span(
        session=session,
        node_id=payload.node_id,
        char_start=payload.char_start,
        char_end=payload.char_end,
        expand=payload.expand,
    )
