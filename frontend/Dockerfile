# syntax=docker/dockerfile:1
ARG UV_VERSION=0.11.6

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:3.11-slim-bookworm

COPY --from=uv /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-cache -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
