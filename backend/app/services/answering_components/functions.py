"""OpenAI function schemas and execution helpers for retrieval-backed answering."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas import RetrievalEvidence, StructuralFilters
from app.services.retrieval_components import (
    BM25Service,
    DenseRetriever,
    HybridRetriever,
    StructuralContentRetriever,
)


def build_retrieval_functions(*, default_limit: int) -> list[dict[str, Any]]:
    """Return OpenAI function schemas for retrieval operations."""

    filter_schema = {
        "type": "object",
        "properties": {
            "part_number": {"type": ["string", "null"]},
            "section_number": {"type": ["string", "null"]},
            "subpart": {"type": ["string", "null"]},
            "marker_path": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "additionalProperties": False,
    }
    search_parameters = {
        "type": "object",
        "properties": {
            "query_text": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": default_limit},
            "filters": filter_schema,
        },
        "required": ["query_text"],
        "additionalProperties": False,
    }
    return [
        _function_schema(
            name="bm25_search",
            description="Lexical search for exact wording, mentions, quotes, and literal checks.",
            parameters=search_parameters,
        ),
        _function_schema(
            name="hybrid_search",
            description="General-purpose retrieval combining lexical and semantic search.",
            parameters=search_parameters,
        ),
        _function_schema(
            name="dense_search",
            description="Vector-only semantic search for broader conceptual similarity.",
            parameters=search_parameters,
        ),
        _function_schema(
            name="lookup_structural_content",
            description="Fetch precomputed structural content such as full section text or outlines.",
            parameters={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "enum": ["section_text", "part_outline", "subpart_outline"],
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
                    "filters": filter_schema,
                },
                "required": ["target"],
                "additionalProperties": False,
            },
        ),
        _function_schema(
            name="get_section_text",
            description="Fetch the full text of one specific HIPAA section by section number.",
            parameters={
                "type": "object",
                "properties": {
                    "section_number": {"type": "string"},
                },
                "required": ["section_number"],
                "additionalProperties": False,
            },
        ),
        _function_schema(
            name="list_part_outline",
            description="Fetch the outline for one HIPAA part, including subparts and sections.",
            parameters={
                "type": "object",
                "properties": {
                    "part_number": {"type": "string"},
                },
                "required": ["part_number"],
                "additionalProperties": False,
            },
        ),
        _function_schema(
            name="list_subpart_outline",
            description="Fetch the outline for one HIPAA subpart.",
            parameters={
                "type": "object",
                "properties": {
                    "subpart": {"type": "string"},
                    "part_number": {"type": ["string", "null"]},
                },
                "required": ["subpart"],
                "additionalProperties": False,
            },
        ),
    ]


@dataclass(slots=True)
class FunctionExecutionResult:
    """Normalized result of one function call."""

    function_name: str
    function_args: dict[str, Any]
    content: str
    evidence: list[RetrievalEvidence]


class RetrievalFunctionExecutor:
    """Execute model function calls against retrieval components."""

    def __init__(
        self,
        *,
        session: AsyncSession,
        bm25_service: BM25Service,
        dense_retriever: DenseRetriever,
        hybrid_retriever: HybridRetriever,
        structural_retriever: StructuralContentRetriever,
        default_limit: int,
    ) -> None:
        self.session = session
        self.bm25_service = bm25_service
        self.dense_retriever = dense_retriever
        self.hybrid_retriever = hybrid_retriever
        self.structural_retriever = structural_retriever
        self.default_limit = default_limit

    async def execute(self, function_name: str, raw_arguments: str) -> FunctionExecutionResult:
        """Execute a function call and return serialized content plus evidence."""

        args = json.loads(raw_arguments or "{}")
        if function_name == "bm25_search":
            evidence = await self.bm25_service.search(
                session=self.session,
                query_text=str(args["query_text"]),
                limit=self._limit(args.get("limit")),
                filters=self._filters(args.get("filters")),
            )
        elif function_name == "hybrid_search":
            evidence = await self.hybrid_retriever.search(
                session=self.session,
                query_text=str(args["query_text"]),
                limit=self._limit(args.get("limit")),
                filters=self._filters(args.get("filters")),
            )
        elif function_name == "dense_search":
            evidence = await self.dense_retriever.search(
                session=self.session,
                query_text=str(args["query_text"]),
                limit=self._limit(args.get("limit")),
                filters=self._filters(args.get("filters")),
            )
        elif function_name == "lookup_structural_content":
            evidence = await self.structural_retriever.lookup(
                session=self.session,
                target=str(args["target"]),
                limit=self._limit(args.get("limit"), upper_bound=10),
                filters=self._filters(args.get("filters")),
            )
        elif function_name == "get_section_text":
            evidence = await self.structural_retriever.lookup(
                session=self.session,
                target="section_text",
                limit=1,
                filters=StructuralFilters(section_number=str(args["section_number"])),
            )
        elif function_name == "list_part_outline":
            evidence = await self.structural_retriever.lookup(
                session=self.session,
                target="part_outline",
                limit=1,
                filters=StructuralFilters(part_number=str(args["part_number"])),
            )
        elif function_name == "list_subpart_outline":
            evidence = await self.structural_retriever.lookup(
                session=self.session,
                target="subpart_outline",
                limit=1,
                filters=StructuralFilters(
                    part_number=_optional_string(args.get("part_number")),
                    subpart=str(args["subpart"]).upper(),
                ),
            )
        else:
            raise ValueError(f"Unsupported function: {function_name}")

        payload = {
            "function_name": function_name,
            "function_args": args,
            "result_count": len(evidence),
            "results": [item.model_dump() for item in evidence],
        }
        return FunctionExecutionResult(
            function_name=function_name,
            function_args=args,
            content=json.dumps(payload, ensure_ascii=True),
            evidence=evidence,
        )

    def _limit(self, value: Any, *, upper_bound: int = 20) -> int:
        """Clamp a requested limit into the allowed range."""

        if value is None:
            return min(self.default_limit, upper_bound)
        return max(1, min(int(value), upper_bound))

    def _filters(self, payload: Any) -> StructuralFilters | None:
        """Validate optional structural filters."""

        if payload in (None, {}):
            return None
        return StructuralFilters.model_validate(payload)


def _function_schema(*, name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Wrap a function schema in the OpenAI function-calling envelope."""

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _optional_string(value: Any) -> str | None:
    """Normalize optional string inputs from model arguments."""

    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
