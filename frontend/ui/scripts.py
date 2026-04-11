"""Client-side JavaScript snippets passed to Gradio ``js=`` handlers."""

from __future__ import annotations

from typing import Final

# State is not reliably passed to client-side ``js=``; a hidden Textbox carries the payload.
COPY_ANSWER_JS: Final[str] = r"""(text) => {
  const raw = Array.isArray(text) ? text[0] : text;
  const s = raw == null ? "" : String(raw);
  if (!s) return;
  const fallback = (str) => {
    const el = document.createElement("textarea");
    el.value = str;
    el.setAttribute("readonly", "");
    el.style.position = "fixed";
    el.style.left = "-9999px";
    document.body.appendChild(el);
    el.select();
    try { document.execCommand("copy"); } catch (e) {}
    document.body.removeChild(el);
  };
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(s).catch(() => fallback(s));
  } else {
    fallback(s);
  }
}"""
