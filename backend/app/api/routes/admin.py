from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db_session
from app.schemas import HealthResponse, SearchRequest, SearchResponse
from app.services.retrieval import RetrievalService


router = APIRouter(tags=["admin"])
DbSessionDep = Annotated[AsyncSession, Depends(get_db_session)]


@router.get("/health")
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/admin/search/bm25")
async def search_bm25(payload: SearchRequest, session: DbSessionDep) -> SearchResponse:
    service = RetrievalService()
    results = await service.bm25_service.search(
        session=session,
        query_text=payload.query_text,
        limit=payload.limit,
        filters=payload.filters,
    )
    return SearchResponse(
        mode="bm25_only",
        query_text=payload.query_text,
        limit=payload.limit,
        filters=payload.filters,
        results=results,
    )


@router.post("/admin/search/dense")
async def search_dense(payload: SearchRequest, session: DbSessionDep) -> SearchResponse:
    service = RetrievalService()
    results = await service.dense_search(
        session=session,
        query_text=payload.query_text,
        limit=payload.limit,
        filters=payload.filters,
    )
    return SearchResponse(
        mode="dense",
        query_text=payload.query_text,
        limit=payload.limit,
        filters=payload.filters,
        results=results,
    )


@router.post("/admin/search/hybrid")
async def search_hybrid(payload: SearchRequest, session: DbSessionDep) -> SearchResponse:
    service = RetrievalService()
    results = await service.hybrid_search(
        session=session,
        query_text=payload.query_text,
        limit=payload.limit,
        filters=payload.filters,
    )
    return SearchResponse(
        mode="hybrid",
        query_text=payload.query_text,
        limit=payload.limit,
        filters=payload.filters,
        results=results,
    )
