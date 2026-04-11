"""Application-specific exceptions and HTTP / WebSocket response mapping."""

from __future__ import annotations


class AppError(Exception):
    """Base class for errors mapped to HTTP JSON and WebSocket close behavior."""

    def __init__(
        self,
        message: str,
        *,
        http_status: int = 500,
        ws_close_code: int = 1011,
    ) -> None:
        """Create an application error with stable status and WebSocket close metadata.

        Args:
            message (str): Safe, user-facing or log-safe description.
            http_status (int): HTTP status for REST :class:`fastapi.responses.JSONResponse`.
            ws_close_code (int): Starlette/WebSocket close code when surfaced over WS.

        Returns:
            None

        Raises:
            None
        """

        super().__init__(message)
        self.message = message
        self.http_status = http_status
        self.ws_close_code = ws_close_code


class ConfigurationError(AppError):
    """Raised when required settings (e.g. API keys) are missing or invalid."""

    def __init__(self, message: str) -> None:
        """Signal misconfiguration (typically HTTP 503).

        Args:
            message (str): Explanation (e.g. missing ``OPENAI_API_KEY``).

        Returns:
            None

        Raises:
            None
        """

        super().__init__(message, http_status=503, ws_close_code=1011)


class ServiceUnavailableError(AppError):
    """Raised when an external dependency fails after configuration is present."""

    def __init__(self, message: str) -> None:
        """Signal upstream outage or quota exhaustion.

        Args:
            message (str): Short description suitable for clients and logs.

        Returns:
            None

        Raises:
            None
        """

        super().__init__(message, http_status=503, ws_close_code=1011)
