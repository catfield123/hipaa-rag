"""Derive WebSocket URLs from the configured HTTP ``BACKEND_URL``."""

from __future__ import annotations

from typing import Final

from .config import BACKEND_URL

_WS_SCHEME_HTTP: Final[str] = "http://"
_WS_SCHEME_HTTPS: Final[str] = "https://"
_WS_TARGET_HTTP: Final[str] = "ws://"
_WS_TARGET_HTTPS: Final[str] = "wss://"


def http_to_ws(base_url: str) -> str:
    """Map an HTTP(S) origin to ws(s):// for WebSocket connections.

    Args:
        base_url (str): Base URL such as ``http://host:port`` or ``https://…``.

    Returns:
        str: Matching ``ws://`` or ``wss://`` URL with the same host/path/query.
            If the scheme is not ``http`` or ``https``, returns ``base_url`` stripped.
    """

    u = base_url.strip().rstrip("/")
    if u.startswith(_WS_SCHEME_HTTPS):
        return _WS_TARGET_HTTPS + u[len(_WS_SCHEME_HTTPS) :]
    if u.startswith(_WS_SCHEME_HTTP):
        return _WS_TARGET_HTTP + u[len(_WS_SCHEME_HTTP) :]
    return u


def rag_ws_url() -> str:
    """Full WebSocket URL for the streaming RAG query endpoint.

    Returns:
        str: ``{ws(s)://}{BACKEND_URL host}/rag/query/ws`` (no trailing slash on host).
    """

    return f"{http_to_ws(BACKEND_URL)}/rag/query/ws"
