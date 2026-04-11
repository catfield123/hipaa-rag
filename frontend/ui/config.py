"""HTTP backend base URL and WebSocket timeouts (environment-driven)."""

from __future__ import annotations

import os
from typing import Final


def _env_int(name: str, default: int) -> int:
    """Parse an optional integer environment variable.

    Args:
        name (str): Variable name (e.g. ``WS_RECV_TIMEOUT_SEC``).
        default (int): Value used when unset or empty.

    Returns:
        int: Parsed integer.

    Raises:
        ValueError: If the value is set but not a valid base-10 integer.
    """

    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw, 10)


BACKEND_URL: Final[str] = os.getenv("BACKEND_URL", "http://backend:8000").strip()
WS_RECV_TIMEOUT_SEC: Final[int] = _env_int("WS_RECV_TIMEOUT_SEC", 600)
