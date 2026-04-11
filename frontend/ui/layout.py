"""Gradio layout: question panel, streaming answer, quotes/sources accordions."""

from __future__ import annotations

import gradio as gr

from .rag_query import run_rag_query
from .rendering import progress_bar_html
from .scripts import COPY_ANSWER_JS


def build_demo() -> gr.Blocks:
    """Construct the HIPAA RAG chat UI (question, output, WebSocket-driven updates).

    Returns:
        gr.Blocks: Configured app; call ``.queue().launch()`` from the entrypoint.
    """

    with gr.Blocks(title="HIPAA RAG") as demo:
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
                    value=progress_bar_html(0, visible=False),
                    elem_id="pipeline-status-html",
                    elem_classes=["pipeline-status-block"],
                )
                output_panel = gr.Markdown(elem_classes=["output-body-scroll", "md-answer-top"])

        with gr.Accordion("Quotes", open=False):
            quotes_box = gr.Markdown(elem_classes=["panel-scroll"])
        with gr.Accordion("Sources", open=False):
            sources_box = gr.Markdown(elem_classes=["panel-scroll"])

        submit.click(
            run_rag_query,
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
            js=COPY_ANSWER_JS,
        )

    return demo
