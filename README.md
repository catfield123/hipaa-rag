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

<p align="center">
  <img src="assets/demonstraion.gif" width=380 />
</p>

## Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [API](#api)
- [Getting started](#getting-started)

## Overview

Regulatory Q&A over an ingested HIPAA text corpus: retrieval combines **BM25** (PostgreSQL `pg_textsearch`) and **dense vectors** (`pgvector`); an **iterative tool-calling** flow (`AnsweringService`) gathers evidence before a grounded final completion.

**Components**

| Layer | Role |
|-------|------|
| Ingest | Chunk normalized markdown, embed with OpenAI, persist rows and indexes. |
| API | FastAPI: RAG via `POST /api/rag/query` and `WebSocket /api/rag/query/ws`, admin search debug routes. |
| UI | Gradio. |
| nginx | Reverse proxy: `/` → UI, `/api/` → backend. |
| Tunnel | Optional ngrok service forwarding to nginx. |

## Architecture

### Ingestion and storage

- **Chunking** (`MarkdownChunker`): single pass over CFR-style markdown; tracks Part → Subpart → § section; splits at § boundaries and paragraph markers `(a)`, `(1)(i)`, etc.; nested markers use a stack; continuation lines attach to the active chunk; boilerplate lines (e.g. table of contents, Federal Register lines, `[Reserved]`) are dropped.
- **Lexical index**: BM25 on a stored generated column `search_text` (path + body) via `pg_textsearch`.
- **Dense storage**: embeddings in `pgvector`; queries use **exact** nearest-neighbor distance (no HNSW/IVFFlat in this deployment).
- **Ingest completion**: BM25 index is dropped if present and rebuilt after load (`CREATE INDEX … USING bm25`).

PDF helpers live in [`backend/app/ingest/pdf_parser.py`](backend/app/ingest/pdf_parser.py). The demo ships with pre-generated [`filtered_markdown.md`](filtered_markdown.md); point `FILTERED_MARKDOWN_SOURCE` at your file or regenerate from PDF using that module.

### Query pipeline

Implementation: `backend/app/services/answering.py` (`AnsweringService`). Pattern: **iterative retrieval** with **mandatory** retrieval tool calls per round, then a **separate** `decide_research_status` call, then optional further rounds up to `agent_max_rounds`.

**Per round**

1. Chat completion with retrieval tools (`tool_choice` required): `hybrid_search`, `bm25_search`, `lookup_structural_content`, `get_section_text`, `list_part_outline`, `list_subpart_outline`.
2. `RetrievalFunctionExecutor` executes tools against PostgreSQL; evidence is merged and deduplicated; history is updated.
3. Chat completion with only `decide_research_status` returns `ResearchDecision` (`continue_retrieval`, `intent`, etc.).

Tool and session wiring runs once at the start of `answer_question`; there is no standalone orchestrator service.

| `continue_retrieval` | Behavior |
|----------------------|----------|
| `false` | Final chat completion over accumulated evidence (optional streaming) → `ChatQueryResponse`. |
| `true`, rounds left | Next iteration from step 1. |
| `true`, `agent_max_rounds` reached | Forced `ResearchDecision`, then final completion as above. |

WebSocket clients may receive `type: status` payloads with `phase` values such as `START`, `PLAN`, `RETRIEVE`, `DECIDE`, `ANSWER`.

## Configuration

Copy `.env.example` to `.env`. Additional knobs (retrieval limits, `agent_max_rounds`, etc.) map from `backend/app/config.py` to `UPPER_SNAKE_CASE` environment variables.

| Variable | Purpose |
|----------|---------|
| `NGROK_AUTHTOKEN` | ngrok authentication (Compose `ngrok` service). |
| `NGROK_URL` | Reserved static hostname (e.g. `*.ngrok-free.app`); passed as `http --url=…`. |
| `OPENAI_API_KEY` | OpenAI API key (chat + embeddings). |
| `OPENAI_CHAT_MODEL` | Chat model id (e.g. `gpt-4.1-mini`). |
| `OPENAI_EMBEDDING_MODEL` | Embedding model id (e.g. `text-embedding-3-large`). |
| `EMBEDDING_DIMENSION` | Vector width; must match the embedding model (e.g. `3072` for `text-embedding-3-large`). |
| `POSTGRES_DB` | Database name. |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` | Credentials; must align with `DATABASE_URL`. |
| `DATABASE_URL` | Async SQLAlchemy URL (`postgresql+asyncpg://…`); in Compose the host is the `db` service. |
| `BACKEND_URL` | Base URL the Gradio process uses for API/WebSocket calls (Compose default: `http://nginx/api`). |
| `FILTERED_MARKDOWN_SOURCE` | Host path to the markdown file mounted into the ingest container as `/data/filtered_markdown.md`. |

## API

All routes below are served behind nginx with path prefix **`/api`** (FastAPI `root_path`). Example base: `http://localhost`.

| Method | Path | Description |
|--------|------|---------------|
| `GET` | `/api/health` | Liveness: `{"status": "ok"}`. |
| `POST` | `/api/rag/query` | Body: `{"question": "…"}`. Returns answer, quotes, sources, intent, `retrieval_rounds`. |
| `WebSocket` | `/api/rag/query/ws` | Client sends one text frame: `{"question": "…"}`; server streams status/delta events, then a result payload aligned with `POST /rag/query`. |
| `POST` | `/api/admin/search/bm25` | Debug: BM25 search over chunks. |
| `POST` | `/api/admin/search/dense` | Debug: dense vector search. |
| `POST` | `/api/admin/search/hybrid` | Debug: hybrid retrieval. |
| `POST` | `/api/admin/search/structure` | Debug: structural lookup; request must include `structure_target`. |

Interactive schema: `/api/docs` (Swagger UI).

## Getting started

Docker Compose is the supported path. Variable reference: [Configuration](#configuration).

### 1. Environment

Run:
```bash
cp .env.example .env
``` 

And set at least `OPENAI_API_KEY`, `DATABASE_URL`, and `FILTERED_MARKDOWN_SOURCE` (path to `filtered_markdown.md` on the host).

### 2. Database

```bash
docker compose up -d db
```

### 3. Ingest (required before the API is useful)

Without rows in `retrieval_chunks`, search and RAG return empty results. The `ingest` service waits for PostgreSQL, runs `alembic upgrade head`, then `python -m app.ingest.run` (chunk, embed, load, rebuild BM25).

```bash
docker compose run --rm ingest
```

Wait for `[ingest] finished successfully`. 

Re-run after you change source markdown, the embedding model, or the DB schema (in dev, often `docker compose down -v`, rebuild and ingest again).

### 4. Application stack

```bash
docker compose up -d backend frontend nginx
```

### 5. Verify

| What | URL |
|-------|-----|
| UI | [http://localhost/](http://localhost/) |
| Health | [http://localhost/api/health](http://localhost/api/health) |
| OpenAPI | [http://localhost/api/docs](http://localhost/api/docs) |

### Optional: HTTPS via ngrok

1. Reserve a static hostname in ngrok; set `NGROK_AUTHTOKEN` and `NGROK_URL` in `.env`.
2. After ingest, include ngrok in the stack:  
   `docker compose up -d db backend frontend nginx ngrok`  
   or `docker compose up -d` if every service is defined in your Compose file.
3. UI: `https://<NGROK_URL>/` · HTTP API: `https://<NGROK_URL>/api/…` · WebSocket from the browser: `wss://<NGROK_URL>/api/rag/query/ws`.

Default `BACKEND_URL=http://nginx/api` stays correct: Gradio calls nginx inside the Compose network; ngrok only fronts nginx for external clients.
