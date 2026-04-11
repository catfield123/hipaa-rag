"""FastAPI application setup for the HIPAA RAG backend."""

from __future__ import annotations

from app.api.routes.admin import router as admin_router
from app.api.routes.rag import router as rag_router
from app.config import get_settings
from app.core.exceptions import AppError
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    root_path=settings.api_root_path,
    description=(
        "HIPAA-focused RAG API: health checks, admin retrieval debugging, and "
        "chat-style querying over ingested regulatory text. "
        "WebSocket endpoint `/rag/query/ws` streams status events then returns the same "
        "JSON shape as `POST /rag/query`."
    ),
)
app.include_router(admin_router)
app.include_router(rag_router)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Map :class:`AppError` to a JSON body and HTTP status.

    Args:
        request (Request): Incoming request (unused; required by FastAPI).
        exc (AppError): Domain or configuration error.

    Returns:
        JSONResponse: ``{\"detail\": ...}`` with the exception's HTTP status code.

    Raises:
        None
    """

    return JSONResponse(status_code=exc.http_status, content={"detail": exc.message})
