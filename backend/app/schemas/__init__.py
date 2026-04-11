"""Shared API and service-layer schemas for the HIPAA RAG backend.

Re-exports chat, planning, retrieval, system, and types symbols for convenience imports
(``from app.schemas import ...``).
"""

from app.schemas.chat import ChatQueryRequest, ChatQueryResponse, QuoteSpan, SourceItem
from app.schemas.planning import ResearchDecision
from app.schemas.retrieval import (
    RetrievalEvidence,
    SearchRequest,
    SearchResponse,
    StructuralFilters,
)
from app.schemas.system import HealthResponse, IngestionResult, IngestionSummary
from app.schemas.types import (
    AgentPipelinePhaseEnum,
    QueryIntentEnum,
    RagWsEventType,
    RetrievalCallSkipReason,
    RetrievalModeEnum,
    StructuralContentTargetEnum,
)

__all__ = [
    "AgentPipelinePhaseEnum",
    "ChatQueryRequest",
    "ChatQueryResponse",
    "HealthResponse",
    "IngestionResult",
    "IngestionSummary",
    "QueryIntentEnum",
    "QuoteSpan",
    "RagWsEventType",
    "ResearchDecision",
    "RetrievalCallSkipReason",
    "RetrievalEvidence",
    "RetrievalModeEnum",
    "SearchRequest",
    "SearchResponse",
    "SourceItem",
    "StructuralContentTargetEnum",
    "StructuralFilters",
]
