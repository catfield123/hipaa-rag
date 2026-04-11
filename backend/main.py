"""ASGI entrypoint re-exporting the FastAPI ``app`` for uvicorn (``main:app``)."""

from app.main import app
