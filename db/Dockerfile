# syntax=docker/dockerfile:1
# PostgreSQL + pgvector base, extended with Timescale pg_textsearch for BM25.
# https://github.com/timescale/pg_textsearch

ARG PG_MAJOR=17
ARG PG_TEXTSEARCH_VERSION=v1.0.0

FROM pgvector/pgvector:pg${PG_MAJOR} AS builder

ARG PG_MAJOR
ARG PG_TEXTSEARCH_VERSION

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        postgresql-server-dev-${PG_MAJOR}; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

RUN git clone --branch "${PG_TEXTSEARCH_VERSION}" --depth 1 https://github.com/timescale/pg_textsearch.git

WORKDIR /tmp/pg_textsearch

RUN make && make install

FROM pgvector/pgvector:pg${PG_MAJOR}

ARG PG_MAJOR

LABEL org.opencontainers.image.title="hipaa-rag-postgres" \
      org.opencontainers.image.description="PostgreSQL with pgvector and pg_textsearch (BM25)"

COPY --from=builder /usr/lib/postgresql/${PG_MAJOR}/lib/pg_textsearch.so /usr/lib/postgresql/${PG_MAJOR}/lib/
COPY --from=builder /usr/share/postgresql/${PG_MAJOR}/extension/pg_textsearch* /usr/share/postgresql/${PG_MAJOR}/extension/
