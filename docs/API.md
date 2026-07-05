# Application API

The API layer exposes the document intelligence workflow over HTTP for demos, integration tests, and later UI work.

## Install Runtime

FastAPI and multipart upload support are declared as an optional project extra:

```bash
conda run -n crypto_env python -m pip install -e ".[api]"
```

In the current `crypto_env`, the API dependencies have been installed directly so the API can run now.

## Start The Server

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.api \
  --host 127.0.0.1 \
  --port 8010
```

OpenAPI docs:

```text
http://127.0.0.1:8010/docs
```

Dashboard:

```text
http://127.0.0.1:8010/
```

Health check:

```bash
curl http://127.0.0.1:8010/health
```

## Dashboard

The dashboard is served by the FastAPI app and calls the same HTTP endpoints as external clients.

It includes:

- Retrieval benchmark summary and a hero strip explaining the platform to first-time visitors.
- Query form with a **retrieval backend picker** (tfidf / dense / hybrid) and a **compare mode** that fires the same question at all three backends and renders the ranked lists side by side, flagging rank positions where a backend disagrees with TF-IDF.
- Agent run form with task, domain, writer model, reasoning model, custom model entry, GPU device, API key, endpoint, and token controls.
- **Agent trace stepper**: each LangGraph node as an expandable card with status, a duration bar, and state summaries, plus an engine badge (`langgraph` / `sequential`).
- **Evaluation tab**: deterministic CI-gated metrics with their floors, the retrieval benchmark, and the committed offline judge + RAGAS snapshot with agreement panels ("three scorers, one story").
- Document upload and index rebuild form (all backends).
- Run history browser for AgentOps traces and MLOps runs.
- Answer, quality, citation, latency, export, and raw JSON panes.
- A cold-start banner for free-tier hosting: the page polls `/health` while the instance wakes and unlocks automatically.

Recommended model setup:

- Writer model: `qwen3_8b_default`
- Writer device: `cuda:1`
- Reasoning model: `deepseek_r1_distill_qwen_14b_reasoning`
- Reasoning device: `cuda:1`
- Enable reasoning planner only for harder questions.

For fast deterministic demos, use `extractive_baseline` as the writer and leave the reasoning model as `None`.

For hosted DeepSeek API demos, either configure `.env` first or paste a session-only key into the dashboard, then choose `deepseek_v4_flash_cloud` or `deepseek_v4_pro_cloud`. See [API_KEYS.md](API_KEYS.md).

## Query RAG

```bash
curl -X POST http://127.0.0.1:8010/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "index_path": "data/processed/vector_index",
    "question": "What does the platform do with documents?",
    "top_k": 3,
    "candidate_k": 10,
    "reranker": "lexical"
  }'
```

The response includes the cited answer, retrieved chunks, backend metadata, reranker metadata, quality metrics, and latency.

## Run The Agent

```bash
curl -X POST http://127.0.0.1:8010/agent/run \
  -H "Content-Type: application/json" \
  -d '{
    "index_path": "data/processed/vector_index",
    "question": "Create a research brief about the document intelligence platform.",
    "task": "brief",
    "domain": "academic",
    "top_k": 3,
    "model_profile": "extractive_baseline"
  }'
```

The response includes the final answer, AgentOps trace events, LLMOps metadata, quality metrics, metrics, and trace file path. It also includes `answer_warning`, which is set when the answer was truncated (for example when a thinking model spent its token budget on hidden reasoning) and is `null` otherwise.

Set `reasoning_effort` (`auto`/`none`/`low`/`medium`/`high`) to control a thinking model's hidden reasoning. `auto` leaves the provider default untouched; `none` disables thinking on Gemini so the whole token budget goes to the visible answer. It is forwarded to OpenAI-compatible endpoints, including dashboard-added models.

List available model profiles:

```bash
curl http://127.0.0.1:8010/model-profiles
```

## Upload A Document

```bash
curl -X POST http://127.0.0.1:8010/documents/upload \
  -F "raw_dir=data/raw" \
  -F "file=@/path/to/document.pdf"
```

Supported upload types are `.pdf`, `.md`, and `.txt`. After upload, rebuild the index before querying the new document.

## List Documents

```bash
curl "http://127.0.0.1:8010/documents?raw_dir=data/raw&index_path=data/processed/vector_index"
```

Lists the supported documents in the raw folder with size, modified time, and an `indexed` flag per file. The response also reports `indexed_but_deleted` (documents still in the index whose raw file is gone) and `index_stale` (true when the raw folder and the index disagree), which the dashboard uses to prompt a rebuild.

## Delete A Document

```bash
curl -X DELETE "http://127.0.0.1:8010/documents/report.pdf?raw_dir=data/raw"
```

Deletes one document from the raw folder. Filenames are sanitized so they cannot escape the raw directory, and only supported document types can be deleted. The index keeps serving the deleted document's chunks until the next rebuild, so the dashboard shows a stale-index hint until you click Rebuild Index.

## Rebuild The Index

```bash
curl -X POST http://127.0.0.1:8010/pipeline/rebuild-index \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "data/raw",
    "chunks_path": "data/processed/chunks.jsonl",
    "index_path": "data/processed/vector_index",
    "backend": "tfidf",
    "chunk_size": 800,
    "chunk_overlap": 120
  }'
```

This endpoint runs ingestion and index building in one call. It is useful for demos and local iteration; production deployments should usually run this as a tracked MLOps job.

`backend` accepts `tfidf`, `dense`, `dense_lsa`, `sentence_transformers`, or `hybrid`. The hybrid backend fuses BM25 and dense rankings with weighted reciprocal-rank fusion and honors two optional fields: `rrf_k` (default 60) and `hybrid_dense_weight` (0–1, default 0.5; BM25 gets the remainder).

## Monitoring Summary

```bash
curl http://127.0.0.1:8010/monitoring/retrieval-benchmark
```

This returns the current retrieval benchmark headline metrics for the dashboard.

## Offline Evaluation Snapshot

```bash
curl http://127.0.0.1:8010/monitoring/offline-eval
```

Returns the committed snapshot of offline evaluations (LLM-as-judge, inter-judge agreement, RAGAS) from `data/reference/offline_eval_snapshot.json`. These evaluations call live models, so they cannot run in the deterministic container build; the latest dated run is committed instead and the dashboard's Evaluation tab labels it accordingly. Returns `{"available": false}` when no snapshot is committed.

## Run History

Recent AgentOps traces:

```bash
curl "http://127.0.0.1:8010/history/agent-traces?limit=10"
```

AgentOps trace detail:

```bash
curl http://127.0.0.1:8010/history/agent-traces/agent_5ccb17b4f24f
```

Recent MLOps runs:

```bash
curl "http://127.0.0.1:8010/history/mlops-runs?limit=10"
```

MLOps run detail:

```bash
curl http://127.0.0.1:8010/history/mlops-runs/run_a2ad04439354
```

The list endpoints return compact summaries for the dashboard. The detail endpoints return the full JSON record plus the same summary fields.
