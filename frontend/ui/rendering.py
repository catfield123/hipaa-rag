"""Markdown/HTML helpers for status lines, citations, and the pipeline progress bar."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def format_status_line(message: str) -> str:
    """Format a retrieval/status message for inline italic display.

    Args:
        message (str): Raw status text from the backend.

    Returns:
        str: Markdown line prefixed with ``*Current:*``.
    """

    return f"*Current:* {message}"


def render_reference_label(item: Mapping[str, Any]) -> str:
    """Pick a human-readable label for a chunk-like dict.

    Args:
        item (Mapping[str, Any]): Quote or source object from the API.

    Returns:
        str: ``path_text`` or first available section/part/subpart, else ``Unknown source``.
    """

    label = (
        item.get("path_text")
        or item.get("section")
        or item.get("part")
        or item.get("subpart")
        or "Unknown source"
    )
    return str(label)


def render_quotes(quotes: list[Mapping[str, Any]]) -> str:
    """Render quote blocks as bold titles plus body text.

    Args:
        quotes (list[Mapping[str, Any]]): Quote payloads (may include ``text``).

    Returns:
        str: Markdown sections separated by blank lines; empty items skipped.
    """

    blocks: list[str] = []
    for item in quotes:
        label = render_reference_label(item)
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        blocks.append(f"**{label}**\n\n{text}")
    return "\n\n".join(blocks)


def render_sources(sources: list[Mapping[str, Any]]) -> str:
    """Render a bullet list of source labels.

    Args:
        sources (list[Mapping[str, Any]]): Source rows from the final RAG result.

    Returns:
        str: Markdown bullet list (one line per source).
    """

    rendered = [f"- {render_reference_label(item)}" for item in sources]
    return "\n".join(rendered)


def progress_bar_html(percent: int, *, visible: bool, label: str = "") -> str:
    """Build minimal HTML for an indeterminate-style progress strip and label.

    Args:
        percent (int): Fill width 0–100 (clamped).
        visible (bool): If False, returns a hidden placeholder div.
        label (str): Optional caption; HTML-escaped for safe insertion.

    Returns:
        str: HTML fragment with classes ``pipeline-progress-*``.
    """

    if not visible:
        return '<div class="pipeline-progress-wrap hidden"></div>'
    pct = max(0, min(100, percent))
    esc = (
        label.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    label_html = f'<div class="pipeline-progress-label">{esc}</div>' if esc else ""
    return (
        f'<div class="pipeline-progress-wrap">'
        f'<div class="pipeline-progress-track">'
        f'<div class="pipeline-progress-fill" style="width:{pct}%;"></div>'
        f"</div>{label_html}</div>"
    )


def status_to_percent(status_index: int) -> int:
    """Map monotonic status event index to a capped progress percentage.

    Args:
        status_index (int): Number of ``status`` events seen (1-based usage expected).

    Returns:
        int: Value between 6 and 92 so the bar never reads as complete before ``result``.
    """

    return min(92, 6 + status_index * 10)
