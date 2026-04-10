"""Administrative and debugging API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db_session
from app.schemas import HealthResponse, SearchRequest, SearchResponse
from app.services.retrieval import RetrievalService

router = APIRouter(tags=["admin"])
DbSessionDep = Annotated[AsyncSession, Depends(get_db_session)]


def get_retrieval_service() -> RetrievalService:
    """Provide the retrieval facade used by admin debugging endpoints."""

    return RetrievalService()


RetrievalServiceDep = Annotated[RetrievalService, Depends(get_retrieval_service)]


@router.get("/health")
async def health() -> HealthResponse:
    """Return a simple health-check response."""

    return HealthResponse(status="ok")


@router.post("/admin/search/bm25")
async def search_bm25(
    payload: SearchRequest,
    session: DbSessionDep,
    retrieval_service: RetrievalServiceDep,
) -> SearchResponse:
    """Run BM25 retrieval for debugging."""

    results = await retrieval_service.bm25_service.search(
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
async def search_dense(
    payload: SearchRequest,
    session: DbSessionDep,
    retrieval_service: RetrievalServiceDep,
) -> SearchResponse:
    """Run dense retrieval for debugging."""

    results = await retrieval_service.dense_search(
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
async def search_hybrid(
    payload: SearchRequest,
    session: DbSessionDep,
    retrieval_service: RetrievalServiceDep,
) -> SearchResponse:
    """Run hybrid retrieval for debugging."""

    results = await retrieval_service.hybrid_search(
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


@router.post("/admin/search/structure")
async def search_structure(
    payload: SearchRequest,
    session: DbSessionDep,
    retrieval_service: RetrievalServiceDep,
) -> SearchResponse:
    """Run direct structural content lookup for debugging."""

    if payload.structure_target is None:
        raise HTTPException(status_code=400, detail="structure_target is required for structure lookup.")

    results = await retrieval_service.lookup_structural_content(
        session=session,
        target=payload.structure_target,
        limit=payload.limit,
        filters=payload.filters,
    )
    return SearchResponse(
        mode="structure_lookup",
        query_text=payload.query_text,
        limit=payload.limit,
        filters=payload.filters,
        results=results,
    )
