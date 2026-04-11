"""Direct structural content retrieval helpers."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import StructuralContent
from app.schemas.retrieval import RetrievalEvidence, StructuralFilters
from app.schemas.types import StructuralContentTargetEnum
from app.services.chunk_contract import (
    build_structural_content_evidence,
    build_structural_content_filter_clauses,
)


class StructuralContentRetriever:
    """Load precomputed section text and outline rows from ``structural_content``."""

    async def lookup(
        self,
        *,
        session: AsyncSession,
        target: StructuralContentTargetEnum,
        limit: int,
        filters: StructuralFilters | None = None,
    ) -> list[RetrievalEvidence]:
        """Return structural content rows for a target type and optional equality filters.

        Args:
            session (AsyncSession): Async SQLAlchemy session.
            target (StructuralContentTargetEnum): Which structural artifact to fetch.
            limit (int): Maximum rows (typically small for outlines).
            filters (StructuralFilters | None): Part/section/subpart filters when applicable.

        Returns:
            list[RetrievalEvidence]: Evidence with ``retrieval_mode`` ``structure_lookup``.

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On database errors.
        """

        rows = (
            await session.execute(
                select(StructuralContent)
                .where(
                    StructuralContent.content_type == target,
                    *build_structural_content_filter_clauses(filters),
                )
                .order_by(StructuralContent.id)
                .limit(limit)
            )
        ).scalars().all()
        return [build_structural_content_evidence(item) for item in rows]
