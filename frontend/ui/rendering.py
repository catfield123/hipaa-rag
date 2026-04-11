def format_status_line(message: str) -> str:
    return f"*Current:* {message}"


def render_reference_label(item: dict) -> str:
    label = item.get("path_text") or item.get("section") or item.get("part") or item.get("subpart") or "Unknown source"
    return str(label)


def render_quotes(quotes: list[dict]) -> str:
    blocks: list[str] = []
    for item in quotes:
        label = render_reference_label(item)
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        blocks.append(f"**{label}**\n\n{text}")
    return "\n\n".join(blocks)


def render_sources(sources: list[dict]) -> str:
    rendered: list[str] = []
    for item in sources:
        rendered.append(f"- {render_reference_label(item)}")
    return "\n".join(rendered)


def progress_bar_html(percent: int, *, visible: bool, label: str = "") -> str:
    if not visible:
        return '<div class="pipeline-progress-wrap hidden"></div>'
    pct = max(0, min(100, percent))
    esc = (
        label.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    label_html = f'<div class="pipeline-progress-label">{esc}</div>' if esc else ""
    return (
        f'<div class="pipeline-progress-wrap">'
        f'<div class="pipeline-progress-track">'
        f'<div class="pipeline-progress-fill" style="width:{pct}%;"></div>'
        f"</div>{label_html}</div>"
    )


def status_to_percent(status_index: int) -> int:
    return min(92, 6 + status_index * 10)
