import os

import gradio as gr
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")


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


def ask_hipaa(question: str) -> tuple[str, str, str]:
    question = question.strip()
    if not question:
        return "", "", "Введите вопрос"

    try:
        response = requests.post(
            f"{BACKEND_URL}/rag/query",
            json={"question": question},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        answer = data.get("answer", "")
        quotes = data.get("quotes", [])
        sources = data.get("sources", [])
        quotes_text = _render_quotes(quotes) if quotes else ""
        sources_text = _render_sources(sources) if sources else ""
        return answer, quotes_text, sources_text
    except Exception as exc:
        return "", "", f"Ошибка запроса: {exc}"


def _run_query(question: str):
    """First yield disables the button only; skip other outputs to avoid heavy Markdown re-renders while waiting."""
    yield (
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.update(interactive=False),
        gr.update(),
        gr.update(),
    )
    answer, quotes_text, sources_text = ask_hipaa(question)
    yield (
        answer,
        quotes_text,
        sources_text,
        gr.update(interactive=True),
        gr.update(visible=True),
        gr.update(visible=True),
    )


with gr.Blocks() as demo:
    gr.Markdown("## HIPAA RAG")
    gr.Markdown("Задавайте вопросы по HIPAA. Интерфейс показывает ответ, цитаты и источники без истории чата.")
    question_input = gr.Textbox(label="Вопрос", placeholder="Например: Does HIPAA mention encryption best practices?")
    answer_box = gr.Markdown(label="Ответ")
    with gr.Accordion("Цитаты", open=False, visible=False) as quotes_accordion:
        quotes_box = gr.Markdown()
    with gr.Accordion("Источники", open=False, visible=False) as sources_accordion:
        sources_box = gr.Markdown()
    submit = gr.Button("Спросить")
    submit.click(
        _run_query,
        inputs=[question_input],
        outputs=[answer_box, quotes_box, sources_box, submit, quotes_accordion, sources_accordion],
        show_progress="hidden",
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
