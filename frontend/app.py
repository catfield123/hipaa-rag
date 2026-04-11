import json
import os
from enum import StrEnum

import gradio as gr
import websocket


class RagWsEventType(StrEnum):
    """Mirror of ``app.schemas.types.RagWsEventType`` (WebSocket ``type`` field)."""

    STATUS = "status"
    ANSWER_DELTA = "answer_delta"
    ERROR = "error"
    RESULT = "result"


BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
_WS_RECV_TIMEOUT_SEC = 600

_UI_CSS = """
.yellow-ask button {
    background: linear-gradient(180deg, #ffe082 0%, #ffca28 100%) !important;
    color: #171717 !important;
    border: 1px solid #f9a825 !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06) !important;
}
.yellow-ask button:hover {
    filter: brightness(0.98);
}
.yellow-ask button:disabled {
    opacity: 0.65 !important;
}
div.panel-scroll {
    max-height: 420px;
    overflow-y: auto;
}
"""


def _http_to_ws(base_url: str) -> str:
    u = base_url.strip().rstrip("/")
    if u.startswith("https://"):
        return "wss://" + u[8:]
    if u.startswith("http://"):
        return "ws://" + u[7:]
    return u


def _rag_ws_url() -> str:
    return f"{_http_to_ws(BACKEND_URL)}/rag/query/ws"


def _format_status_line(message: str) -> str:
    return f"*Current:* {message}"


def _render_reference_label(item: dict) -> str:
    label = item.get("path_text") or item.get("section") or item.get("part") or item.get("subpart") or "Unknown source"
    return str(label)


def _render_quotes(quotes: list[dict]) -> str:
    blocks: list[str] = []
    for item in quotes:
        label = _render_reference_label(item)
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        blocks.append(f"**{label}**\n\n{text}")
    return "\n\n".join(blocks)


def _render_sources(sources: list[dict]) -> str:
    rendered: list[str] = []
    for item in sources:
        rendered.append(f"- {_render_reference_label(item)}")
    return "\n".join(rendered)


def _run_query(question: str):
    """Right panel: latest status only (replacing); switches to streamed answer on first delta; statuses stop updating."""
    yield ("", "", "", gr.update(interactive=False))
    q = (question or "").strip()
    if not q:
        yield ("*Enter a question on the left and press Ask.*", "", "", gr.update(interactive=True))
        return

    ws: websocket.WebSocket | None = None
    try:
        ws = websocket.WebSocket()
        ws.connect(_rag_ws_url(), timeout=30)
        ws.settimeout(_WS_RECV_TIMEOUT_SEC)
        ws.send(json.dumps({"question": q}))

        yield ("*Waiting…*", "", "", gr.update(interactive=False))

        answer_mode = False
        accumulated_answer = ""

        while True:
            raw = ws.recv()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            msg = json.loads(raw)
            mtype = msg.get("type")
            if mtype == RagWsEventType.STATUS:
                if not answer_mode:
                    line = _format_status_line(str(msg.get("message") or ""))
                    yield (line, gr.skip(), gr.skip(), gr.update(interactive=False))
            elif mtype == RagWsEventType.ANSWER_DELTA:
                answer_mode = True
                accumulated_answer += str(msg.get("text") or "")
                yield (accumulated_answer, gr.skip(), gr.skip(), gr.update(interactive=False))
            elif mtype == RagWsEventType.RESULT:
                answer = str(msg.get("answer") or "")
                quotes = msg.get("quotes") or []
                sources = msg.get("sources") or []
                quotes_text = _render_quotes(quotes) if quotes else ""
                sources_text = _render_sources(sources) if sources else ""
                yield (answer, quotes_text, sources_text, gr.update(interactive=True))
                return
            elif mtype == RagWsEventType.ERROR:
                err = str(msg.get("message") or "error")
                yield (f"**Error:** {err}", "", "", gr.update(interactive=True))
                return
    except TimeoutError:
        yield ("**Error:** response timed out.", "", "", gr.update(interactive=True))
    except Exception as exc:
        yield (f"**Error:** {exc}", "", "", gr.update(interactive=True))
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass


with gr.Blocks(css=_UI_CSS, title="HIPAA RAG") as demo:
    gr.Markdown("## HIPAA RAG")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            question_input = gr.Textbox(
                label="Question",
                lines=10,
                placeholder="e.g. What does HIPAA require for encryption of ePHI?",
                elem_id="question-box",
            )
            submit = gr.Button("Ask", elem_classes=["yellow-ask"])

        with gr.Column(scale=1):
            gr.Markdown("### Output")
            output_panel = gr.Markdown(elem_classes=["panel-scroll"])

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Quotes")
            quotes_box = gr.Markdown(elem_classes=["panel-scroll"])
        with gr.Column(scale=1):
            gr.Markdown("### Sources")
            sources_box = gr.Markdown(elem_classes=["panel-scroll"])

    submit.click(
        _run_query,
        inputs=[question_input],
        outputs=[output_panel, quotes_box, sources_box, submit],
        show_progress="hidden",
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
