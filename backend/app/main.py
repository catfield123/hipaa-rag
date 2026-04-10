"""FastAPI application setup for the HIPAA RAG backend."""

from fastapi import FastAPI

from app.api.routes.admin import router as admin_router
from app.api.routes.chat import router as chat_router
from app.config import get_settings

settings = get_settings()
app = FastAPI(title=settings.app_name, root_path=settings.api_root_path)
app.include_router(admin_router)
app.include_router(chat_router)
