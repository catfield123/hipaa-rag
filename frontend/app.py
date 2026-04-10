import json
import os

import gradio as gr
import websocket

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
# Long-running retrieval + LLM; `connect` timeout is separate (seconds).
_WS_RECV_TIMEOUT_SEC = 600


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
    # path_text already ends with the section trail; do not append markers (they duplicate the tail).
    label = item.get("path_text") or item.get("section") or item.get("part") or item.get("subpart") or "Unknown source"
    return str(label)


def _render_quotes(quotes: list[dict]) -> str:
    blocks: list[str] = []
    for item in quotes:
        label = _render_reference_label(item)
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        blocks.append(f"**{label}**\n{text}")
    return "\n\n".join(blocks)


def _render_sources(sources: list[dict]) -> str:
    rendered: list[str] = []
    for item in sources:
        rendered.append(f"- {_render_reference_label(item)}")
    return "\n".join(rendered)


def _run_query(question: str):
    """Stream agent status into the answer field via WebSocket; final yield fills answer, quotes, sources."""
    yield (
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.update(interactive=False),
    )
    q = (question or "").strip()
    if not q:
        yield ("", "", "Enter a question.", gr.update(interactive=True))
        return

    ws: websocket.WebSocket | None = None
    try:
        ws = websocket.WebSocket()
        ws.connect(_rag_ws_url(), timeout=30)
        ws.settimeout(_WS_RECV_TIMEOUT_SEC)
        ws.send(json.dumps({"question": q}))
        accumulated_answer = ""
        while True:
            raw = ws.recv()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            msg = json.loads(raw)
            mtype = msg.get("type")
            if mtype == "status":
                line = _format_status_line(str(msg.get("message") or ""))
                yield (line, gr.skip(), gr.skip(), gr.update(interactive=False))
            elif mtype == "answer_delta":
                accumulated_answer += str(msg.get("text") or "")
                yield (accumulated_answer, gr.skip(), gr.skip(), gr.update(interactive=False))
            elif mtype == "result":
                answer = str(msg.get("answer") or "")
                quotes = msg.get("quotes") or []
                sources = msg.get("sources") or []
                quotes_text = _render_quotes(quotes) if quotes else ""
                sources_text = _render_sources(sources) if sources else ""
                yield (answer, quotes_text, sources_text, gr.update(interactive=True))
                return
            elif mtype == "error":
                err = str(msg.get("message") or "error")
                yield ("", "", f"Error: {err}", gr.update(interactive=True))
                return
    except TimeoutError:
        yield ("", "", "Response timed out.", gr.update(interactive=True))
    except Exception as exc:
        yield ("", "", f"Request error: {exc}", gr.update(interactive=True))
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass


with gr.Blocks() as demo:
    gr.Markdown("## HIPAA RAG")
    gr.Markdown(
        "Ask questions about HIPAA. This UI shows the answer, quotes, and sources without chat history."
    )
    question_input = gr.Textbox(
        label="Question",
        placeholder="e.g. Does HIPAA mention encryption best practices?",
    )
    submit = gr.Button("Ask")
    answer_box = gr.Markdown(label="Answer")
    with gr.Accordion("Quotes", open=False):
        quotes_box = gr.Markdown()
    with gr.Accordion("Sources", open=False):
        sources_box = gr.Markdown()
    submit.click(
        _run_query,
        inputs=[question_input],
        outputs=[answer_box, quotes_box, sources_box, submit],
        show_progress="hidden",
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
