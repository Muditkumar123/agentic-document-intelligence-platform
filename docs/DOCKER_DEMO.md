# Docker Demo

This demo runs the FastAPI dashboard in a container for a clean, repeatable project walkthrough.

## Start The API

From the project root:

```bash
docker compose -f infra/docker/docker-compose.yml up --build adip-api
```

Open:

```text
http://127.0.0.1:8010/
```

OpenAPI docs:

```text
http://127.0.0.1:8010/docs
```

## Demo Flow

1. Open the `Index` tab.
2. Upload a `.pdf`, `.md`, or `.txt` document.
3. Click `Rebuild Index`.
4. Open the `RAG` tab and ask a cited question.
5. Open the `Agent` tab for a research brief or QA answer.
6. Review the quality metrics, citations, trace, and raw JSON.
7. Click `Export` to download a Markdown report.

The container mounts the local `data/`, `config/`, and `prompts/` folders, so uploaded documents, processed chunks, indexes, and monitoring traces remain visible on the host machine.

## Run Smoke Tests

```bash
docker compose -f infra/docker/docker-compose.yml run --rm adip-smoke
```

## Local LLM Note

The Docker demo is intentionally lightweight and CPU-oriented. Use the host `crypto_env` workflow for local CUDA model runs with Qwen or DeepSeek:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.api \
  --host 127.0.0.1 \
  --port 8010
```

In the dashboard, choose `qwen3_8b_default` as the writer model and `deepseek_r1_distill_qwen_14b_reasoning` as the reasoning model when you want the stronger local LLM demo.
