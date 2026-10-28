# Deployment

The project ships as a single self-contained container that serves the FastAPI API and the dashboard, with a demo index built in. It needs **no GPU and no model download** — the default serving path is the deterministic extractive writer over a TF-IDF index.

## Run locally (one command)

```bash
docker compose -f infra/docker/docker-compose.yml up --build
# then open http://localhost:8010
```

Or with plain Docker:

```bash
docker build -t adip -f infra/docker/Dockerfile .
docker run --rm -p 8010:8010 adip
```

Check it's healthy:

```bash
curl -fsS http://localhost:8010/health
# {"status":"ok","service":"agentic-document-intelligence-api","version":"0.1.0","write_access":"local_only","max_upload_mb":20.0}
```

## Image design

- **Multi-stage build** ([infra/docker/Dockerfile](../infra/docker/Dockerfile)): a builder installs pinned runtime deps into a venv; the runtime stage copies only that venv plus the files needed to serve. No `dev`/test deps and no `torch`/`transformers` in the image.
- **Pinned dependencies** ([infra/docker/requirements.lock.txt](../infra/docker/requirements.lock.txt)) for reproducible builds, on a pinned `python:3.14-slim-bookworm` base.
- **Non-root** runtime user (`appuser`, uid 10001).
- **Healthcheck** hits `/health`; deploy platforms use the same path.
- **Demo index baked in**: the build runs ingestion → retrieval eval → generation eval over the real public-document corpus (`data/eval/`), so the dashboard answers questions and shows Answer-Quality metrics immediately. The [entrypoint](../infra/docker/entrypoint.sh) rebuilds the index on first start if a mounted volume shadows it.
- **`$PORT` aware**: the entrypoint binds `${PORT:-8010}`, so Render/Fly/Railway can inject their own port.

## Deploy to a host

### Render (one click, free tier)

The repo includes a [`render.yaml`](../render.yaml) blueprint. In the Render dashboard: **New + → Blueprint** → point it at this repo. Render builds the Dockerfile, injects `$PORT`, health-checks `/health`, and generates a random `ADIP_ADMIN_TOKEN`. `autoDeploy` redeploys on every push to `main`. Copy the token from the service's **Environment** page into the dashboard's Admin Token field to upload, delete, or rebuild; visitors without it can still query and run the agent.

The production instance is live at **https://agentic-document-intelligence.onrender.com** (free tier: spins down after ~15 minutes idle; the first request after that takes up to a minute).

### Fly.io

```bash
fly launch --dockerfile infra/docker/Dockerfile --internal-port 8010
fly deploy
```

### Railway / any Docker host

Point it at `infra/docker/Dockerfile`, expose the port the platform provides via `$PORT`, and set the health check path to `/health`.

## Configuration

- `PORT` — port to bind (default `8010`).
- `ADIP_ADMIN_TOKEN` — bearer token for the write endpoints (upload, delete, rebuild). Without it those endpoints refuse every request that doesn't come from loopback, which inside a container means every request. See the security model in [API.md](API.md#security-model).
- `ADIP_MAX_UPLOAD_MB` — largest accepted document upload (default `20`).
- `ADIP_RATE_LIMIT_PER_MINUTE` — requests per client per minute on the query, agent, model-check and write endpoints (default `60`, `0` turns it off). Render's proxy hides client addresses, so there it is one shared budget.
- `ADIP_ALLOW_MODEL_DOWNLOAD` — set to `1` to let the server fetch embedding or reranker models it doesn't have cached. Off by default.
- `ADIP_RERANKER_ONNX` — path to an exported `model.onnx` (or `model.int8.onnx`). Cross-encoder reranking for the model it was exported from then runs on ONNX Runtime on CPU whenever a request asks for `"reranker": "cross_encoder"` with that model (`/rag/query`, `/agent/run`, the RAG CLIs and evals); other reranker models stay on PyTorch. Needs the `onnx` extra. Export once with `python -c "from pathlib import Path; from adip.rag.onnx_reranker import export_cross_encoder; export_cross_encoder('cross-encoder/ms-marco-MiniLM-L-6-v2', Path('data/models/onnx-reranker'))"`. Export with the Hugging Face id, not a local cache path: the id recorded in `onnx_reranker.json` is what requests are matched against. Unset by default.
- Hosted-LLM writers (Gemini/DeepSeek/Groq) are optional and **session-only**: API keys are supplied per request from the dashboard and never written to disk or baked into the image. The default extractive writer needs no keys.

## Persistence (optional)

By default the container is stateless: the baked demo index is used and uploaded documents live only for the container's lifetime. To persist uploads and a rebuilt index across restarts, mount a volume at `/app/data` (uncomment the `volumes:` block in the compose file); the entrypoint rebuilds the index on first start.

## CI verification

The CI pipeline's `docker` job builds this image on every push/PR and smoke-tests `/health`, the dashboard, and `/monitoring/generation-eval`, so the deploy path can't silently break. See [CICD.md](CICD.md).
