from pathlib import Path

_FRONTEND_ROOT = Path(__file__).resolve().parent.parent
APP_CSS = (_FRONTEND_ROOT / "static" / "app.css").read_text(encoding="utf-8")
