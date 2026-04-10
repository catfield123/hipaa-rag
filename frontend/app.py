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


def ask_hipaa(question: str, history: list[list[str]], include_debug: bool) -> tuple[list[list[str]], str, str]:
    history = history or []
    question = question.strip()
    if not question:
        return history, "", "Введите вопрос"

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

        extras: list[str] = []
        if quotes:
            extras.append(_render_quotes(quotes))
        if sources:
            extras.append(_render_sources(sources))
        if include_debug and data.get("debug"):
            extras.append(f"Debug:\n```json\n{json.dumps(data['debug'], indent=2)}\n```")

        rendered = answer
        if extras:
            rendered = f"{answer}\n\n" + "\n\n".join(extras)

        history = history + [[question, rendered]]
        return history, "", ""
    except Exception as exc:
        return history, question, f"Ошибка запроса: {exc}"


with gr.Blocks() as demo:
    gr.Markdown("## HIPAA RAG")
    gr.Markdown("Задавайте вопросы по HIPAA, а ответ вернется с цитатами и источниками.")
    chatbot = gr.Chatbot(label="Диалог", height=500)
    error_box = gr.Markdown()
    question_input = gr.Textbox(label="Вопрос", placeholder="Например: Does HIPAA mention encryption best practices?")
    debug_checkbox = gr.Checkbox(label="Показать debug retrieval", value=False)
    submit = gr.Button("Спросить")
    submit.click(
        ask_hipaa,
        inputs=[question_input, chatbot, debug_checkbox],
        outputs=[chatbot, question_input, error_box],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
