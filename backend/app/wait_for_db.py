"""Block until PostgreSQL accepts connections (used by Docker ``depends_on`` health gates)."""

from __future__ import annotations

import time

import psycopg
from app.config import get_settings


def main() -> None:
    """Poll the database DSN until a connection succeeds or retries are exhausted.

    Args:
        None

    Returns:
        None: Exits normally once the server accepts TCP + auth.

    Raises:
        RuntimeError: If no connection succeeds within the fixed retry budget.
    """

    settings = get_settings()
    dsn = settings.psycopg_connect_url
    last_error: Exception | None = None

    print("[wait_for_db] Polling PostgreSQL until a connection succeeds...", flush=True)

    for attempt in range(30):
        try:
            with psycopg.connect(dsn, connect_timeout=3):
                print("[wait_for_db] PostgreSQL is accepting connections.", flush=True)
                return
        except Exception as exc:  # pragma: no cover - operational retry path
            last_error = exc
            print(
                f"[wait_for_db] attempt {attempt + 1}/30 failed: {exc!s}; retrying in 1s...",
                flush=True,
            )
            time.sleep(1)

    raise RuntimeError("Database did not become ready in time.") from last_error


if __name__ == "__main__":
    main()
