# Docker Demo

This demo runs the FastAPI dashboard in a container for a clean, repeatable project walkthrough. For the production image design and one-click hosting (Render/Fly), see [DEPLOY.md](DEPLOY.md).

## Start The API

From the project root:

```bash
docker compose -f infra/docker/docker-compose.yml up --build
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
4. Manage the corpus in the `Raw Documents` list: each file shows its size and an indexed/not-indexed badge, with a Delete button. A stale-index hint appears whenever the raw folder and the index disagree.
5. Open the `RAG` tab and ask a cited question.
6. Open the `Agent` tab for a research brief or QA answer.
7. Review the quality metrics, citations, trace, and raw JSON.
8. Click `Export` to download a Markdown report.

The image ships with a demo index already built from the real public-document corpus (`data/eval/`), so retrieval, cited answers, and the Answer-Quality tiles work immediately — no setup needed. The container is stateless by default; see [DEPLOY.md](DEPLOY.md#persistence-optional) to persist uploads across restarts. The Docker image is built and smoke-tested in CI.

## Local LLM Note

The Docker demo is intentionally lightweight and CPU-oriented. Use the host `crypto_env` workflow for local CUDA model runs with Qwen or DeepSeek:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.api \
  --host 127.0.0.1 \
  --port 8010
```

In the dashboard, choose `qwen3_8b_default` as the writer model and `deepseek_r1_distill_qwen_14b_reasoning` as the reasoning model when you want the stronger local LLM demo.
