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
# {"status":"ok","service":"agentic-document-intelligence-api","version":"0.1.0"}
```

## Image design

- **Multi-stage build** ([infra/docker/Dockerfile](../infra/docker/Dockerfile)): a builder installs pinned runtime deps into a venv; the runtime stage copies only that venv plus the files needed to serve. No `dev`/test deps and no `torch`/`transformers` in the image.
- **Pinned dependencies** ([infra/docker/requirements.lock.txt](../infra/docker/requirements.lock.txt)) for reproducible builds, on a pinned `python:3.10-slim-bookworm` base.
- **Non-root** runtime user (`appuser`, uid 10001).
- **Healthcheck** hits `/health`; deploy platforms use the same path.
- **Demo index baked in**: the build runs ingestion → retrieval eval → generation eval over the real public-document corpus (`data/eval/`), so the dashboard answers questions and shows Answer-Quality metrics immediately. The [entrypoint](../infra/docker/entrypoint.sh) rebuilds the index on first start if a mounted volume shadows it.
- **`$PORT` aware**: the entrypoint binds `${PORT:-8010}`, so Render/Fly/Railway can inject their own port.

## Deploy to a host

### Render (one click, free tier)

The repo includes a [`render.yaml`](../render.yaml) blueprint. In the Render dashboard: **New + → Blueprint** → point it at this repo. Render builds the Dockerfile, injects `$PORT`, and health-checks `/health`. `autoDeploy` redeploys on every push to `main`.

### Fly.io

```bash
fly launch --dockerfile infra/docker/Dockerfile --internal-port 8010
fly deploy
```

### Railway / any Docker host

Point it at `infra/docker/Dockerfile`, expose the port the platform provides via `$PORT`, and set the health check path to `/health`.

## Configuration

- `PORT` — port to bind (default `8010`).
- Hosted-LLM writers (Gemini/DeepSeek/Groq) are optional and **session-only**: API keys are supplied per request from the dashboard and never written to disk or baked into the image. The default extractive writer needs no keys.

## Persistence (optional)

By default the container is stateless: the baked demo index is used and uploaded documents live only for the container's lifetime. To persist uploads and a rebuilt index across restarts, mount a volume at `/app/data` (uncomment the `volumes:` block in the compose file); the entrypoint rebuilds the index on first start.

## CI verification

The CI pipeline's `docker` job builds this image on every push/PR and smoke-tests `/health`, the dashboard, and `/monitoring/generation-eval`, so the deploy path can't silently break. See [CICD.md](CICD.md).
