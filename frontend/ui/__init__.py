"""Gradio UI package: layout, WebSocket RAG client, styles, and theme."""

from __future__ import annotations

from .layout import build_demo
from .styles import APP_CSS

__all__ = ["APP_CSS", "build_demo"]
