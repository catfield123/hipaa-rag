"""Administrative and debugging API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db_session
from app.schemas import HealthResponse, SearchRequest, SearchResponse
from app.services.retrieval_components import BM25Service, DenseRetriever, HybridRetriever, StructuralContentRetriever
from app.services.retrieval_components.dependencies import (
    get_bm25_service,
    get_dense_retriever,
    get_hybrid_retriever,
    get_structural_content_retriever,
)

router = APIRouter(tags=["admin"])
DbSessionDep = Annotated[AsyncSession, Depends(get_db_session)]


Bm25ServiceDep = Annotated[BM25Service, Depends(get_bm25_service)]
DenseRetrieverDep = Annotated[DenseRetriever, Depends(get_dense_retriever)]
HybridRetrieverDep = Annotated[HybridRetriever, Depends(get_hybrid_retriever)]
StructuralRetrieverDep = Annotated[
    StructuralContentRetriever,
    Depends(get_structural_content_retriever),
]


@router.get("/health")
async def health() -> HealthResponse:
    """Return a simple health-check response."""

    return HealthResponse(status="ok")


@router.post("/admin/search/bm25")
async def search_bm25(
    payload: SearchRequest,
    session: DbSessionDep,
    bm25_service: Bm25ServiceDep,
) -> SearchResponse:
    """Run BM25 retrieval for debugging."""

    results = await bm25_service.search(
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
    dense_retriever: DenseRetrieverDep,
) -> SearchResponse:
    """Run dense retrieval for debugging."""

    results = await dense_retriever.search(
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
    hybrid_retriever: HybridRetrieverDep,
) -> SearchResponse:
    """Run hybrid retrieval for debugging."""

    results = await hybrid_retriever.search(
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
    structural_retriever: StructuralRetrieverDep,
) -> SearchResponse:
    """Run direct structural content lookup for debugging."""

    if payload.structure_target is None:
        raise HTTPException(status_code=400, detail="structure_target is required for structure lookup.")

    results = await structural_retriever.lookup(
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
