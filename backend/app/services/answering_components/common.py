"""Shared helpers for the answering pipeline."""

from __future__ import annotations

import json
import re

from app.config import Settings
from app.schemas import EvidenceDecision, QueryPlan, QueryVariant, RetrievalEvidence


def build_evidence_payload(
    evidence: list[RetrievalEvidence],
    *,
    limit: int,
) -> list[dict[str, object]]:
    """Convert evidence objects into a compact JSON-friendly payload."""

    return [
        {
            "chunk_id": item.chunk_id,
            "path_text": item.path_text,
            "part": item.part,
            "subpart": item.subpart,
            "section": item.section,
            "markers": item.markers,
            "retrieval_mode": item.retrieval_mode,
            "score": item.score,
            "text": item.text,
            "metadata": item.metadata,
        }
        for item in evidence[:limit]
    ]


def sanitize_query_plan(plan: QueryPlan, *, settings: Settings) -> QueryPlan | None:
    """Normalize and deduplicate a query plan."""

    queries = sanitize_query_variants(plan.queries, settings=settings)
    if not queries:
        return None
    return plan.model_copy(update={"queries": queries})


def sanitize_evidence_decision(
    decision: EvidenceDecision,
    *,
    settings: Settings,
) -> EvidenceDecision:
    """Normalize the follow-up queries attached to an evidence decision."""

    if decision.sufficient:
        return decision.model_copy(update={"next_queries": []})
    return decision.model_copy(
        update={"next_queries": sanitize_query_variants(decision.next_queries, settings=settings)}
    )


def sanitize_query_variants(
    queries: list[QueryVariant],
    *,
    settings: Settings,
) -> list[QueryVariant]:
    """Clean and deduplicate query variants while enforcing planner limits."""

    sanitized: list[QueryVariant] = []
    seen: set[str] = set()
    for query in queries:
        text = re.sub(r"\s+", " ", query.text).strip()
        if not text:
            continue
        if query.mode == "structure_lookup" and query.structure_target is None:
            continue

        filters_payload = query.filters.model_dump() if query.filters else None
        signature = json.dumps(
            {
                "text": text.lower(),
                "mode": query.mode,
                "structure_target": query.structure_target,
                "filters": filters_payload,
            },
            sort_keys=True,
        )
        if signature in seen:
            continue

        seen.add(signature)
        sanitized.append(query.model_copy(update={"text": text}))
        if len(sanitized) >= settings.query_rewrite_limit:
            break
    return sanitized
