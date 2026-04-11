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
/* Hidden payload for Copy — no layout footprint (value still sent to the browser for JS) */
.copy-payload-hidden {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    overflow: hidden !important;
}
/* Equal-height chat columns: shared min height, stack grows inside */
.col-chat {
    min-height: 520px;
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
}
.col-chat .fill {
    flex: 1 1 auto;
    min-height: 0;
}
/* Right column: stack Output → progress bar → status/answer tight at the top (no giant flex gap) */
.col-out {
    align-items: stretch !important;
}
.col-out .out-heading {
    flex: 0 0 auto !important;
}
.col-out .out-heading h1,
.col-out .out-heading h2,
.col-out .out-heading h3,
.col-out .out-heading p {
    margin: 0 0 0.35rem 0 !important;
}
.col-out #pipeline-status-html,
.col-out .pipeline-status-block {
    flex: 0 0 auto !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}
.col-out .output-body-scroll {
    flex: 0 1 auto !important;
    min-height: 0 !important;
}
/* Answer markdown: top-aligned, full width (no vertical centering in the panel) */
.md-answer-top {
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
    align-items: stretch !important;
    text-align: left !important;
}
.md-answer-top > div {
    width: 100% !important;
    justify-content: flex-start !important;
}
.md-answer-top .prose,
.md-answer-top [class*="markdown"] {
    text-align: left !important;
    width: 100%;
}
div.panel-scroll {
    max-height: 580px;
    overflow-y: auto;
    width: 100%;
}
/* Question column only: keep a comfortable typing area */
div.panel-scroll.question-panel {
    min-height: 420px;
}
/* Answer panel: scroll long answers, no forced min-height (keeps status flush under progress bar) */
div.output-body-scroll {
    max-height: 580px;
    overflow-y: auto;
    width: 100%;
}
div.output-body-scroll.md-answer-top {
    justify-content: flex-start !important;
    align-items: stretch !important;
    align-content: flex-start !important;
}
/* Native-style pipeline bar (not a slider) */
.pipeline-progress-wrap {
    width: 100%;
    margin-top: 0 !important;
    margin-bottom: 0.35rem !important;
}
.pipeline-progress-wrap.hidden {
    display: none !important;
}
.pipeline-progress-track {
    height: 8px;
    border-radius: 4px;
    background: #e8e8e8;
    overflow: hidden;
}
.pipeline-progress-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #ffca28, #ffb300);
    transition: width 0.2s ease;
}
.pipeline-progress-label {
    font-size: 0.78rem;
    color: #555;
    margin-top: 0.25rem;
}
"""

# State is not reliably passed to client-side `js=`; hidden Textbox value is.
_COPY_ANSWER_JS = r"""(text) => {
  const raw = Array.isArray(text) ? text[0] : text;
  const s = raw == null ? "" : String(raw);
  if (!s) return;
  const fallback = (str) => {
    const el = document.createElement("textarea");
    el.value = str;
    el.setAttribute("readonly", "");
    el.style.position = "fixed";
    el.style.left = "-9999px";
    document.body.appendChild(el);
    el.select();
    try { document.execCommand("copy"); } catch (e) {}
    document.body.removeChild(el);
  };
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(s).catch(() => fallback(s));
  } else {
    fallback(s);
  }
}"""


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
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        blocks.append(f"**{label}**\n\n{text}")
    return "\n\n".join(blocks)


def _render_sources(sources: list[dict]) -> str:
    rendered: list[str] = []
    for item in sources:
        rendered.append(f"- {_render_reference_label(item)}")
    return "\n".join(rendered)


def _progress_bar_html(percent: int, *, visible: bool, label: str = "") -> str:
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


def _status_to_percent(status_index: int) -> int:
    return min(92, 6 + status_index * 10)


def _run_query(question: str):
    """Stream status/answer; progress bar until first answer token; quotes/sources in accordions."""
    empty_payload = gr.update(value="")
    empty_bar = _progress_bar_html(0, visible=False)
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
        ws.connect(_rag_ws_url(), timeout=30)
        ws.settimeout(_WS_RECV_TIMEOUT_SEC)
        ws.send(json.dumps({"question": q}))

        yield (
            "*Waiting…*",
            empty_payload,
            _progress_bar_html(4, visible=True, label="Connecting…"),
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
            msg = json.loads(raw)
            mtype = msg.get("type")
            if mtype == RagWsEventType.STATUS:
                if not answer_mode:
                    status_count += 1
                    line = _format_status_line(str(msg.get("message") or ""))
                    pct = _status_to_percent(status_count)
                    yield (
                        line,
                        gr.skip(),
                        _progress_bar_html(pct, visible=True, label="Retrieval / planning…"),
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
                    _progress_bar_html(0, visible=False),
                    gr.skip(),
                    gr.skip(),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                )
            elif mtype == RagWsEventType.RESULT:
                answer = str(msg.get("answer") or "")
                quotes = msg.get("quotes") or []
                sources = msg.get("sources") or []
                quotes_text = _render_quotes(quotes) if quotes else ""
                sources_text = _render_sources(sources) if sources else ""
                yield (
                    answer,
                    gr.update(value=answer),
                    _progress_bar_html(0, visible=False),
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
                    _progress_bar_html(0, visible=False),
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
            _progress_bar_html(0, visible=False),
            "",
            "",
            gr.update(interactive=True),
            gr.update(interactive=False),
        )
    except Exception as exc:
        yield (
            f"**Error:** {exc}",
            empty_payload,
            _progress_bar_html(0, visible=False),
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


with gr.Blocks(
    title="HIPAA RAG",
    theme=gr.themes.Default(primary_hue="yellow"),
) as demo:
    gr.Markdown("## HIPAA RAG")

    with gr.Row(equal_height=True, elem_classes=["row-chat"]):
        with gr.Column(scale=1, elem_classes=["col-chat"]):
            question_input = gr.Textbox(
                label="Question",
                lines=14,
                placeholder="e.g. What does HIPAA require for encryption of ePHI?",
                elem_id="question-box",
                elem_classes=["fill", "panel-scroll", "question-panel"],
            )
            submit = gr.Button("Ask", variant="primary")
            answer_copy_payload = gr.Textbox(
                value="",
                visible=False,
                show_label=False,
                lines=1,
                max_lines=1,
                elem_classes=["copy-payload-hidden"],
            )
            copy_answer_btn = gr.Button("Copy answer", variant="secondary", interactive=False)

        with gr.Column(scale=1, elem_classes=["col-chat", "col-out"]):
            gr.Markdown("### Output", elem_classes=["out-heading"])
            pipeline_bar = gr.HTML(
                value=_progress_bar_html(0, visible=False),
                elem_id="pipeline-status-html",
                elem_classes=["pipeline-status-block"],
            )
            output_panel = gr.Markdown(elem_classes=["output-body-scroll", "md-answer-top"])

    with gr.Accordion("Quotes", open=False):
        quotes_box = gr.Markdown(elem_classes=["panel-scroll"])
    with gr.Accordion("Sources", open=False):
        sources_box = gr.Markdown(elem_classes=["panel-scroll"])

    submit.click(
        _run_query,
        inputs=[question_input],
        outputs=[
            output_panel,
            answer_copy_payload,
            pipeline_bar,
            quotes_box,
            sources_box,
            submit,
            copy_answer_btn,
        ],
        show_progress="hidden",
    )

    copy_answer_btn.click(
        fn=None,
        inputs=[answer_copy_payload],
        outputs=None,
        js=_COPY_ANSWER_JS,
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=_UI_CSS,
    )
