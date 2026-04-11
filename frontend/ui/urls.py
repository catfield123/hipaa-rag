from .config import BACKEND_URL


def http_to_ws(base_url: str) -> str:
    u = base_url.strip().rstrip("/")
    if u.startswith("https://"):
        return "wss://" + u[8:]
    if u.startswith("http://"):
        return "ws://" + u[7:]
    return u


def rag_ws_url() -> str:
    return f"{http_to_ws(BACKEND_URL)}/rag/query/ws"
