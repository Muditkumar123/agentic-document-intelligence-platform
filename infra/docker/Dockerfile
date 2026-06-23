# syntax=docker/dockerfile:1
# Production image for the Agentic Document Intelligence Platform API + dashboard.
# Multi-stage: a builder installs pinned runtime deps into a venv, the runtime stage
# copies only that venv + the files needed to serve, builds a demo index, and runs
# as a non-root user. No dev/test deps, no torch/transformers (the default serving
# path is the deterministic extractive writer over a TF-IDF index).

# ---------- builder ----------
FROM python:3.10-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app
RUN python -m venv /opt/venv

# Install pinned runtime dependencies first (cached layer), then the package itself.
COPY infra/docker/requirements.lock.txt ./
RUN pip install -r requirements.lock.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

# ---------- runtime ----------
FROM python:3.10-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8010 \
    PATH="/opt/venv/bin:$PATH"

# poppler-utils: pdftotext for PDF uploads. curl: container healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY config ./config
COPY prompts ./prompts
COPY data/eval ./data/eval
COPY data/reference ./data/reference
COPY infra/docker/entrypoint.sh ./entrypoint.sh

# Build a demo index + eval metrics so the dashboard works out of the box, create a
# non-root user, and hand it ownership of the writable app tree.
RUN chmod +x ./entrypoint.sh \
    && python -m adip.mlops.run_ingestion --input data/eval/raw \
        --output data/processed/chunks.jsonl \
        --metrics-output data/monitoring/ingestion_metrics.json \
    && python -m adip.mlops.run_rag_eval --chunks data/processed/chunks.jsonl \
        --index data/processed/vector_index --golden data/eval/golden_qa.jsonl \
        --backend tfidf --no-faiss --metrics-output data/monitoring/rag_eval_metrics.json \
    && python -m adip.mlops.run_generation_eval --index data/processed/vector_index \
        --golden data/eval/golden_qa.jsonl --abstention-threshold 0.10 \
        --metrics-output data/monitoring/generation_eval_metrics.json \
        --report-output data/monitoring/generation_eval_report.json \
    && mkdir -p data/raw data/monitoring/agent_traces data/monitoring/mlops_runs \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

ENTRYPOINT ["./entrypoint.sh"]
