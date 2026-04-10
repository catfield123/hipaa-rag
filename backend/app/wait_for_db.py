from __future__ import annotations

import time

import psycopg

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    dsn = settings.psycopg_connect_url
    last_error: Exception | None = None

    for _ in range(30):
        try:
            with psycopg.connect(dsn, connect_timeout=3):
                return
        except Exception as exc:  # pragma: no cover - operational retry path
            last_error = exc
            time.sleep(1)

    raise RuntimeError("Database did not become ready in time.") from last_error


if __name__ == "__main__":
    main()
