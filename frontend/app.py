import json
import os

import gradio as gr
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")


def _render_reference_label(item: dict) -> str:
    label = item.get("path_text") or item.get("section") or item.get("part") or item.get("subpart") or "Unknown source"
    markers = item.get("markers") or []
    if markers:
        return f"{label} {' '.join(markers)}"
    return str(label)


def _render_quotes(quotes: list[dict]) -> str:
    rendered: list[str] = []
    for item in quotes:
        label = _render_reference_label(item)
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        rendered.append(f'- "{text}" ({label})')
    return "Quotes:\n" + "\n".join(rendered)


def _render_sources(sources: list[dict]) -> str:
    rendered: list[str] = []
    for item in sources:
        rendered.append(f"- {_render_reference_label(item)}")
    return "Sources:\n" + "\n".join(rendered)


def ask_hipaa(question: str, include_debug: bool) -> tuple[str, str, str, str]:
    question = question.strip()
    if not question:
        return "", "", "", "Введите вопрос"

    try:
        response = requests.post(
            f"{BACKEND_URL}/chat/query",
            json={"question": question, "include_debug": include_debug},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        answer = data.get("answer", "")
        quotes = data.get("quotes", [])
        sources = data.get("sources", [])
        quotes_text = _render_quotes(quotes) if quotes else ""
        sources_text = _render_sources(sources) if sources else ""
        debug_text = ""
        if include_debug and data.get("debug"):
            debug_text = f"```json\n{json.dumps(data['debug'], indent=2)}\n```"
        return answer, quotes_text, sources_text, debug_text
    except Exception as exc:
        return "", "", "", f"Ошибка запроса: {exc}"


with gr.Blocks() as demo:
    gr.Markdown("## HIPAA RAG")
    gr.Markdown("Задавайте вопросы по HIPAA. Интерфейс показывает ответ, цитаты и источники без истории чата.")
    question_input = gr.Textbox(label="Вопрос", placeholder="Например: Does HIPAA mention encryption best practices?")
    debug_checkbox = gr.Checkbox(label="Показать debug retrieval", value=False)
    answer_box = gr.Markdown(label="Ответ")
    quotes_box = gr.Markdown(label="Цитаты")
    sources_box = gr.Markdown(label="Источники")
    debug_box = gr.Markdown(label="Debug")
    submit = gr.Button("Спросить")
    submit.click(
        ask_hipaa,
        inputs=[question_input, debug_checkbox],
        outputs=[answer_box, quotes_box, sources_box, debug_box],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
