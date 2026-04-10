"""Direct structural content retrieval helpers."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import StructuralContent
from app.schemas.retrieval import RetrievalEvidence, StructuralFilters
from app.schemas.types import StructuralContentTarget
from app.services.chunk_contract import (
    build_structural_content_evidence,
    build_structural_content_filter_clauses,
)


class StructuralContentRetriever:
    """Retrieve precomputed section and outline content."""

    async def lookup(
        self,
        *,
        session: AsyncSession,
        target: StructuralContentTarget,
        limit: int,
        filters: StructuralFilters | None = None,
    ) -> list[RetrievalEvidence]:
        """Return structural content matching the requested target and filters."""

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
