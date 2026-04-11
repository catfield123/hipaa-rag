"""OpenAI function schemas and execution helpers for retrieval-backed answering."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.string_templates import errors
from app.schemas.retrieval import RetrievalEvidence, StructuralFilters
from app.schemas.types import StructuralContentTargetEnum
from app.services.retrieval_components import (
    BM25Service,
    DenseRetriever,
    HybridRetriever,
    StructuralContentRetriever,
)


def build_retrieval_functions(*, default_limit: int) -> list[dict[str, Any]]:
    """Build OpenAI ``tools`` schemas for hybrid, BM25, dense, and structural lookups.

    Args:
        default_limit (int): Default ``limit`` injected into search parameter schemas (clamped in executor).

    Returns:
        list[dict[str, Any]]: Function definitions for ``chat.completions`` tool calling.

    Raises:
        None
    """

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
            name="hybrid_search",
            description="Default search for nearly all non-structural questions, especially permissions, requirements, obligations, disclosures, and exceptions.",
            parameters=search_parameters,
        ),
        _function_schema(
            name="bm25_search",
            description="Use only for exact wording, quotes, literal mentions, and explicit verification of whether a phrase appears.",
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
                        "enum": [target.value for target in StructuralContentTargetEnum],
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
    """Normalized result of executing one model-issued retrieval tool call.

    Args (fields):
        function_name (str): Tool name (e.g. ``hybrid_search``).
        function_args (dict[str, Any]): Parsed JSON arguments from the model.
        content (str): JSON string summarizing evidence for prompt history.
        evidence (list[RetrievalEvidence]): Structured hits returned by the retriever.
    """

    function_name: str
    function_args: dict[str, Any]
    content: str
    evidence: list[RetrievalEvidence]


class RetrievalFunctionExecutor:
    """Dispatch OpenAI function calls to BM25, hybrid, dense, or structural retrievers."""

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
        """Wire retriever dependencies and default limits for tool execution.

        Args:
            session (AsyncSession): Database session shared for one answering run.
            bm25_service (BM25Service): Lexical search backend.
            dense_retriever (DenseRetriever): Dense vector backend.
            hybrid_retriever (HybridRetriever): Hybrid fusion backend.
            structural_retriever (StructuralContentRetriever): Structural rows backend.
            default_limit (int): Default hit cap when the model omits ``limit``.

        Returns:
            None

        Raises:
            None
        """

        self.session = session
        self.bm25_service = bm25_service
        self.dense_retriever = dense_retriever
        self.hybrid_retriever = hybrid_retriever
        self.structural_retriever = structural_retriever
        self.default_limit = default_limit

    async def execute(self, function_name: str, raw_arguments: str) -> FunctionExecutionResult:
        """Parse arguments and run the matching retrieval function.

        Args:
            function_name (str): Model-selected tool name.
            raw_arguments (str): JSON object string from the tool call.

        Returns:
            FunctionExecutionResult: Evidence list plus serialized payload for history.

        Raises:
            json.JSONDecodeError: If ``raw_arguments`` is not valid JSON.
            ValueError: If ``function_name`` is not a supported tool.
        """

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
                target=StructuralContentTargetEnum(str(args["target"])),
                limit=self._limit(args.get("limit"), upper_bound=10),
                filters=self._filters(args.get("filters")),
            )
        elif function_name == "get_section_text":
            evidence = await self.structural_retriever.lookup(
                session=self.session,
                target=StructuralContentTargetEnum.SECTION_TEXT,
                limit=1,
                filters=StructuralFilters(section_number=str(args["section_number"])),
            )
        elif function_name == "list_part_outline":
            evidence = await self.structural_retriever.lookup(
                session=self.session,
                target=StructuralContentTargetEnum.PART_OUTLINE,
                limit=1,
                filters=StructuralFilters(part_number=str(args["part_number"])),
            )
        elif function_name == "list_subpart_outline":
            evidence = await self.structural_retriever.lookup(
                session=self.session,
                target=StructuralContentTargetEnum.SUBPART_OUTLINE,
                limit=1,
                filters=StructuralFilters(
                    part_number=_optional_string(args.get("part_number")),
                    subpart=str(args["subpart"]).upper(),
                ),
            )
        else:
            raise ValueError(errors.UNSUPPORTED_RETRIEVAL_FUNCTION.format(function_name=function_name))

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
        """Clamp a model-provided ``limit`` to ``[1, upper_bound]`` with a service default.

        Args:
            value (Any): Raw limit from tool arguments (may be ``None``).
            upper_bound (int): Maximum allowed hits for this tool family.

        Returns:
            int: Safe integer limit.

        Raises:
            None
        """

        if value is None:
            return min(self.default_limit, upper_bound)
        return max(1, min(int(value), upper_bound))

    def _filters(self, payload: Any) -> StructuralFilters | None:
        """Parse optional structural filters from tool arguments.

        Args:
            payload (Any): ``filters`` sub-object or ``None`` / empty dict.

        Returns:
            StructuralFilters | None: Validated filters, or ``None`` when absent.

        Raises:
            ValidationError: If the payload does not match :class:`~app.schemas.retrieval.StructuralFilters`.
        """

        if payload in (None, {}):
            return None
        return StructuralFilters.model_validate(payload)


def _function_schema(*, name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Wrap name, description, and JSON-schema parameters in OpenAI's function tool envelope.

    Args:
        name (str): Exposed function name for the model.
        description (str): Short instruction for when to use the tool.
        parameters (dict[str, Any]): JSON Schema for the ``parameters`` field.

    Returns:
        dict[str, Any]: One entry suitable for the ``tools`` array.

    Raises:
        None
    """

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _optional_string(value: Any) -> str | None:
    """Normalize optional string tool arguments (empty strings become ``None``).

    Args:
        value (Any): Raw argument value.

    Returns:
        str | None: Stripped string, or ``None``.

    Raises:
        None
    """

    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
