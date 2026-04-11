"""Load the bundled CSS file shipped next to the Gradio app."""

from __future__ import annotations

from pathlib import Path
from typing import Final

_FRONTEND_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
APP_CSS: Final[str] = (_FRONTEND_ROOT / "static" / "app.css").read_text(encoding="utf-8")
