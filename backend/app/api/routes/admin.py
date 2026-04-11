"""Administrative and debugging API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.deps import (
    Bm25ServiceDep,
    DbSessionDep,
    DenseRetrieverDep,
    HybridRetrieverDep,
    StructuralRetrieverDep,
)
from app.schemas.retrieval import SearchRequest, SearchResponse
from app.schemas.system import HealthResponse
from app.schemas.types import RetrievalModeEnum

router = APIRouter(tags=["admin"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns `{\"status\": \"ok\"}` when the API process is running.",
)
async def health() -> HealthResponse:
    """Return a simple health-check response.

    Args:
        None

    Returns:
        HealthResponse: Fixed liveness payload.

    Raises:
        None
    """

    return HealthResponse(status="ok")


@router.post(
    "/admin/search/bm25",
    response_model=SearchResponse,
    summary="Debug: BM25 search",
    description="Full-text BM25 search over ingested chunks (pg text search).",
)
async def search_bm25(
    payload: SearchRequest,
    session: DbSessionDep,
    bm25_service: Bm25ServiceDep,
) -> SearchResponse:
    """Run BM25 retrieval for debugging.

    Args:
        payload (SearchRequest): Query text, limit, and optional structural filters.
        session (AsyncSession): Database session.
        bm25_service (BM25Service): Lexical retriever.

    Returns:
        SearchResponse: Mode `bm25_only` and ranked evidence rows.

    Raises:
        None
    """

    results = await bm25_service.search(
        session=session,
        query_text=payload.query_text,
        limit=payload.limit,
        filters=payload.filters,
    )
    return SearchResponse(
        mode=RetrievalModeEnum.BM25_ONLY,
        query_text=payload.query_text,
        limit=payload.limit,
        filters=payload.filters,
        results=results,
    )


@router.post(
    "/admin/search/dense",
    response_model=SearchResponse,
    summary="Debug: dense vector search",
    description="Nearest-neighbor search using pgvector embeddings for the query string.",
)
async def search_dense(
    payload: SearchRequest,
    session: DbSessionDep,
    dense_retriever: DenseRetrieverDep,
) -> SearchResponse:
    """Run dense retrieval for debugging.

    Args:
        payload (SearchRequest): Query text, limit, and optional filters.
        session (AsyncSession): Database session.
        dense_retriever (DenseRetriever): Embedding-backed retriever.

    Returns:
        SearchResponse: Mode `dense` and ranked evidence rows.

    Raises:
        None
    """

    results = await dense_retriever.search(
        session=session,
        query_text=payload.query_text,
        limit=payload.limit,
        filters=payload.filters,
    )
    return SearchResponse(
        mode=RetrievalModeEnum.DENSE,
        query_text=payload.query_text,
        limit=payload.limit,
        filters=payload.filters,
        results=results,
    )


@router.post(
    "/admin/search/hybrid",
    response_model=SearchResponse,
    summary="Debug: hybrid search",
    description="Fusion of dense and BM25 rankings (RRF-style blend used by the retriever).",
)
async def search_hybrid(
    payload: SearchRequest,
    session: DbSessionDep,
    hybrid_retriever: HybridRetrieverDep,
) -> SearchResponse:
    """Run hybrid retrieval for debugging.

    Args:
        payload (SearchRequest): Query text, limit, and optional filters.
        session (AsyncSession): Database session.
        hybrid_retriever (HybridRetriever): Hybrid retriever.

    Returns:
        SearchResponse: Mode `hybrid` and ranked evidence rows.

    Raises:
        None
    """

    results = await hybrid_retriever.search(
        session=session,
        query_text=payload.query_text,
        limit=payload.limit,
        filters=payload.filters,
    )
    return SearchResponse(
        mode=RetrievalModeEnum.HYBRID,
        query_text=payload.query_text,
        limit=payload.limit,
        filters=payload.filters,
        results=results,
    )


@router.post(
    "/admin/search/structure",
    response_model=SearchResponse,
    summary="Debug: structural lookup",
    description=(
        "Direct lookup of predefined structural artifacts (section text, outlines). "
        "Requires `structure_target` in the body."
    ),
    responses={
        400: {
            "description": "`structure_target` missing",
            "content": {
                "application/json": {
                    "example": {"detail": "structure_target is required for structure lookup."}
                }
            },
        },
    },
)
async def search_structure(
    payload: SearchRequest,
    session: DbSessionDep,
    structural_retriever: StructuralRetrieverDep,
) -> SearchResponse:
    """Run direct structural content lookup for debugging.

    Args:
        payload (SearchRequest): Must include `structure_target` for this endpoint.
        session (AsyncSession): Database session.
        structural_retriever (StructuralContentRetriever): Structural backend.

    Returns:
        SearchResponse: Mode `structure_lookup` and rows from structural retrieval.

    Raises:
        HTTPException: If `structure_target` is absent (400).
    """

    if payload.structure_target is None:
        raise HTTPException(status_code=400, detail="structure_target is required for structure lookup.")

    results = await structural_retriever.lookup(
        session=session,
        target=payload.structure_target,
        limit=payload.limit,
        filters=payload.filters,
    )
    return SearchResponse(
        mode=RetrievalModeEnum.STRUCTURE_LOOKUP,
        query_text=payload.query_text,
        limit=payload.limit,
        filters=payload.filters,
        results=results,
    )
