<h1 align="center">HIPAA RAG</h1>

<p align="center">
<a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.11+-34D058?logo=python&logoColor=white" alt="Python 3.11+">
</a>
<a href="https://fastapi.tiangolo.com/">
    <img src="https://img.shields.io/badge/FastAPI-API-34D058?logo=fastapi&logoColor=white" alt="FastAPI">
</a>
<a href="https://www.postgresql.org/">
    <img src="https://img.shields.io/badge/PostgreSQL-database-34D058?logo=postgresql&logoColor=white" alt="PostgreSQL">
</a>
<a href="https://github.com/pgvector/pgvector">
    <img src="https://img.shields.io/badge/pgvector-extension-34D058?logo=postgresql&logoColor=white" alt="pgvector">
</a>
<a href="https://www.docker.com/">
    <img src="https://img.shields.io/badge/Docker-Compose-34D058?logo=docker&logoColor=white" alt="Docker">
</a>
<a href="https://www.gradio.app/">
    <img src="https://img.shields.io/badge/Gradio-UI-34D058?logo=gradio&logoColor=white" alt="Gradio">
</a>
<a href="https://nginx.org/">
    <img src="https://img.shields.io/badge/nginx-reverse_proxy-34D058?logo=nginx&logoColor=white" alt="nginx">
</a>
<a href="https://platform.openai.com/docs/overview">
    <img src="https://img.shields.io/badge/OpenAI-API-34D058?logo=openai&logoColor=white" alt="OpenAI">
</a>
<a href="https://github.com/astral-sh/uv">
    <img src="https://img.shields.io/badge/uv-packages-34D058?logo=uv&logoColor=white" alt="uv">
</a>
<a href="https://ngrok.com/">
    <img src="https://img.shields.io/badge/ngrok-optional-34D058?logo=ngrok&logoColor=white" alt="ngrok">
</a>
</p>

## Problem

Answer questions about HIPAA regulatory text using evidence from an ingested corpus: retrieve relevant passages, combine lexical (BM25) and semantic (vector) search, and produce answers with an LLM agent rather than unconstrained free-form generation.

## Solution

- **Ingest**: Normalized markdown is split into chunks; each chunk is embedded (OpenAI) and stored in PostgreSQL with **pgvector** and **pg_textsearch** (BM25 / full-text).
- **Backend**: FastAPI — health checks, debug search routes, and RAG via `POST /rag/query` plus streaming `WebSocket /rag/query/ws` (status and answer deltas; final payload matches `POST /rag/query`).
- **Frontend**: Gradio UI behind **nginx** (`/` → UI, `/api/` → backend).
- **Public access**: Optional **ngrok** container tunneling nginx (HTTP and WebSocket).

### Chunking and indexes

Chunking is **structure-aware**: a line-by-line pass over normalized HIPAA/CFR-style markdown tracks **Part → Subpart → § section** headings and builds retrieval chunks at § boundaries and at CFR-style paragraph markers such as `(a)` or `(1)(i)`. Nested markers are tracked with a small stack so each chunk carries the correct path; continuation lines are merged into the active chunk when they clearly belong to the same paragraph or list item. Noise lines (e.g. “Contents”, Federal Register references, `[Reserved]`) are skipped.

**Lexical** search uses a **BM25** index over a persisted `search_text` column (path plus body) via **pg_textsearch**. **Dense** vectors live in **pgvector**, but no **HNSW** (or IVFFlat) index is created: the chunk count is small enough that exact nearest-neighbor search is acceptable and avoids extra index build and tuning.

## Environment variables (from `.env.example`)

Copy `.env.example` to `.env` and fill in values.

| Variable | Purpose |
|----------|---------|
| `NGROK_AUTHTOKEN` | ngrok auth token (required by the `ngrok` service in Compose). |
| `NGROK_URL` | Static ngrok domain (e.g. `your-name.ngrok-free.app`) — passed to ngrok as `http --url=...`. |
| `OPENAI_API_KEY` | OpenAI API key for chat and embeddings. |
| `OPENAI_CHAT_MODEL` | Chat model for answers and agent steps (example default: `gpt-4.1-mini`). |
| `OPENAI_EMBEDDING_MODEL` | Embedding model id (example default: `text-embedding-3-large`). |
| `EMBEDDING_DIMENSION` | Vector dimension; must match the embedding model (often `3072` for `text-embedding-3-large`). |
| `POSTGRES_DB` | PostgreSQL database name. |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` | PostgreSQL credentials (must match `DATABASE_URL`). |
| `DATABASE_URL` | Async SQLAlchemy URL (`postgresql+asyncpg://...`); in Docker the DB host is the `db` service name (see `.env.example`). |
| `BACKEND_URL` | Base URL the frontend uses to reach the API (Docker example: `http://nginx/api` so the UI calls the API through the same host as the browser). |
| `FILTERED_MARKDOWN_SOURCE` | **Host** path to the markdown file for ingest; mounted in the container as `/data/filtered_markdown.md`. |

Extra tuning (retriever limits, agent rounds, etc.) is defined in `backend/app/config.py` and can be overridden with the same names in `UPPER_SNAKE_CASE` environment variables without editing code.

## API (behind nginx, base prefix `/api`)

Full paths from the site root: `http://<host>/api/...`. Local (no ngrok): `http://localhost/api/...`.

| Method & path | Description |
|---------------|-------------|
| `GET /api/health` | Liveness: `{ "status": "ok" }`. |
| `POST /api/rag/query` | JSON body `{ "question": "..." }` — full RAG response (quotes, sources, intent, etc.). |
| `WebSocket /api/rag/query/ws` | Send one text frame with JSON `{ "question": "..." }`; receive status/delta events then the final result. |
| `POST /api/admin/search/bm25` | Debug: BM25 over chunks. |
| `POST /api/admin/search/dense` | Debug: vector search. |
| `POST /api/admin/search/hybrid` | Debug: hybrid dense + BM25. |
| `POST /api/admin/search/structure` | Debug: structural lookup; body must include `structure_target`. |

OpenAPI docs: [http://localhost/api/docs](http://localhost/api/docs) (Swagger UI).

## Ingest (run once before relying on RAG)

Run ingest **before** you expect meaningful RAG answers: until chunks and embeddings are loaded, search and the agent operate on an empty corpus.

The repo includes PDF-to-markdown helpers in [`backend/app/ingest/pdf_parser.py`](backend/app/ingest/pdf_parser.py). For a straightforward demo, the output of that parsing step is already checked in as `filtered_markdown.md` (see `FILTERED_MARKDOWN_SOURCE`); you can inspect or adapt the parser there if you need to regenerate markdown from a PDF.

The Docker `ingest` service: wait for PostgreSQL → `alembic upgrade head` → `python -m app.ingest.run` (chunking, embeddings, DB load, indexes).

**Recommended order (local Docker):**

1. Start the database only:  
   `docker compose up -d db`
2. Ensure `.env` has correct `DATABASE_URL`, `OPENAI_*`, `EMBEDDING_DIMENSION`, and `FILTERED_MARKDOWN_SOURCE` pointing at your `filtered_markdown.md`.
3. Run ingest once:  
   `docker compose run --rm ingest`  
   Wait for `[ingest] finished successfully`.
4. Start the rest (no public tunnel):  
   `docker compose up -d backend frontend nginx`

Re-run `ingest` after changing source markdown, embedding model, or DB schema (in development you often recreate the DB and ingest again).

## Local development (Docker)

1. Create `.env` from `.env.example`.
2. `docker compose up -d db`
3. `docker compose run --rm ingest`
4. `docker compose up -d backend frontend nginx`
5. UI: [http://localhost/](http://localhost/) · health check: [http://localhost/api/health](http://localhost/api/health) · interactive API docs (OpenAPI / Swagger): [http://localhost/api/docs](http://localhost/api/docs)

## Internet-facing deploy (ngrok)

1. Register a static domain in ngrok and set `NGROK_AUTHTOKEN` and `NGROK_URL` in `.env`.
2. Run ingest as above, then start the full stack **including** ngrok:  
   `docker compose up -d`  
   or explicitly:  
   `docker compose up -d db backend frontend nginx ngrok`
3. External base URL: `https://<your-ngrok-host>/` for the UI and `https://<your-ngrok-host>/api/...` for HTTP; WebSocket from outside: `wss://<your-ngrok-host>/api/rag/query/ws`.

With the default Compose setup, `BACKEND_URL=http://nginx/api` is correct: the Gradio app opens the RAG WebSocket from **inside** the frontend container to `nginx`, so you normally do not need to change `BACKEND_URL` when adding ngrok.
