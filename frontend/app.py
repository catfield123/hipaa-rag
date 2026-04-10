import os

import gradio as gr
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")


def send_name(name: str) -> str:
    name = name.strip()
    if not name:
        return "Введите имя"

    try:
        response = requests.get(f"{BACKEND_URL}/hello/{name}", timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("greeting", "")
    except Exception as exc:
        return f"Ошибка запроса: {exc}"


with gr.Blocks() as demo:
    gr.Markdown("## Hello UI")
    name_input = gr.Textbox(label="Имя")
    output = gr.Textbox(label="Ответ сервера", interactive=False)
    submit = gr.Button("Отправить")
    submit.click(send_name, inputs=name_input, outputs=output)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
