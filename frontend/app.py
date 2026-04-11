from ui import APP_CSS, build_demo
from ui.theme import LAUNCH_THEME

demo = build_demo()

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=APP_CSS,
        theme=LAUNCH_THEME,
    )
