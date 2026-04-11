"""HIPAA RAG Gradio entrypoint: builds the UI and serves it on port 7860."""

from __future__ import annotations

from ui import APP_CSS, build_demo
from ui.theme import LAUNCH_THEME

demo = build_demo()


def main() -> None:
    """Queue requests and start the Gradio HTTP server (Docker / local)."""

    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=APP_CSS,
        theme=LAUNCH_THEME,
    )


if __name__ == "__main__":
    main()
