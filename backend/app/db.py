"""Database engine and session dependencies for the backend."""

from collections.abc import AsyncIterator

from app.config import get_settings
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

settings = get_settings()

engine: AsyncEngine = create_async_engine(
    settings.database_url,
    future=True,
    pool_pre_ping=True,
)
SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Yield an async SQLAlchemy session for request-scoped dependencies.

    Args:
        None

    Yields:
        AsyncSession: Session bound to :data:`SessionLocal`; closed after the request or WS handler.

    Raises:
        None
    """

    async with SessionLocal() as session:
        yield session
