"""Cross-cutting concerns: configuration errors, shared exception types."""

from app.core.exceptions import AppError, ConfigurationError, ServiceUnavailableError

__all__ = [
    "AppError",
    "ConfigurationError",
    "ServiceUnavailableError",
]
