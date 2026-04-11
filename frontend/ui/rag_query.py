"""Stream RAG answers over WebSocket and map events to Gradio component updates."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any, cast

import gradio as gr
import websocket

from .config import WS_RECV_TIMEOUT_SEC
from .rendering import (
    format_status_line,
    progress_bar_html,
    render_quotes,
    render_sources,
    status_to_percent,
)
from .urls import rag_ws_url
from .ws_types import RagWsEventType

# Seven Gradio outputs: answer MD, hidden copy payload, pipeline HTML, quotes, sources, submit, copy btn.
RagQueryYield = tuple[Any, Any, Any, Any, Any, Any, Any]


def run_rag_query(question: str) -> Iterator[RagQueryYield]:
    """Stream status and answer tokens from the backend RAG WebSocket.

    Yields tuples for: main answer markdown, hidden copy text, pipeline HTML, quotes markdown,
    sources markdown, submit button state, copy button state.

    Args:
        question (str): User question (may be empty).

    Returns:
        Iterator[RagQueryYield]: Progressive UI updates; stops after ``result`` or ``error``.
        Failures are surfaced as markdown in the first tuple element instead of raising.
    """

    empty_payload = gr.update(value="")
    empty_bar = progress_bar_html(0, visible=False)
    yield (
        "",
        empty_payload,
        empty_bar,
        "",
        "",
        gr.update(interactive=False),
        gr.update(interactive=False),
    )
    q = (question or "").strip()
    if not q:
        yield (
            "*Enter a question on the left and press Ask.*",
            empty_payload,
            empty_bar,
            "",
            "",
            gr.update(interactive=True),
            gr.update(interactive=False),
        )
        return

    ws: websocket.WebSocket | None = None
    status_count = 0
    try:
        ws = websocket.WebSocket()
        ws.connect(rag_ws_url(), timeout=30)
        ws.settimeout(WS_RECV_TIMEOUT_SEC)
        ws.send(json.dumps({"question": q}))

        yield (
            "*Waiting…*",
            empty_payload,
            progress_bar_html(4, visible=True, label="Connecting…"),
            "",
            "",
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

        answer_mode = False
        accumulated_answer = ""

        while True:
            raw = ws.recv()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            try:
                msg: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                yield (
                    "**Error:** invalid message from server.",
                    empty_payload,
                    progress_bar_html(0, visible=False),
                    "",
                    "",
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                )
                return
            mtype = msg.get("type")
            if mtype == RagWsEventType.STATUS:
                if not answer_mode:
                    status_count += 1
                    line = format_status_line(str(msg.get("message") or ""))
                    pct = status_to_percent(status_count)
                    yield (
                        line,
                        gr.skip(),
                        progress_bar_html(pct, visible=True, label="Retrieval / planning…"),
                        gr.skip(),
                        gr.skip(),
                        gr.update(interactive=False),
                        gr.update(interactive=False),
                    )
            elif mtype == RagWsEventType.ANSWER_DELTA:
                answer_mode = True
                accumulated_answer += str(msg.get("text") or "")
                yield (
                    accumulated_answer,
                    gr.skip(),
                    progress_bar_html(0, visible=False),
                    gr.skip(),
                    gr.skip(),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                )
            elif mtype == RagWsEventType.RESULT:
                answer = str(msg.get("answer") or "")
                quotes = msg.get("quotes") or []
                sources = msg.get("sources") or []
                quotes_text = (
                    render_quotes(cast(list[Mapping[str, Any]], quotes)) if quotes else ""
                )
                sources_text = (
                    render_sources(cast(list[Mapping[str, Any]], sources)) if sources else ""
                )
                yield (
                    answer,
                    gr.update(value=answer),
                    progress_bar_html(0, visible=False),
                    quotes_text,
                    sources_text,
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                )
                return
            elif mtype == RagWsEventType.ERROR:
                err = str(msg.get("message") or "error")
                yield (
                    f"**Error:** {err}",
                    empty_payload,
                    progress_bar_html(0, visible=False),
                    "",
                    "",
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                )
                return
    except TimeoutError:
        yield (
            "**Error:** response timed out.",
            empty_payload,
            progress_bar_html(0, visible=False),
            "",
            "",
            gr.update(interactive=True),
            gr.update(interactive=False),
        )
    except Exception as exc:
        yield (
            f"**Error:** {exc}",
            empty_payload,
            progress_bar_html(0, visible=False),
            "",
            "",
            gr.update(interactive=True),
            gr.update(interactive=False),
        )
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
