# Agentic Document Intelligence Platform

[![CI](https://github.com/Muditkumar123/agentic-document-intelligence-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/Muditkumar123/agentic-document-intelligence-platform/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Live_Demo-onrender.com-2dd4bf?logo=render&logoColor=white)](https://agentic-document-intelligence.onrender.com)

**Live demo:** [agentic-document-intelligence.onrender.com](https://agentic-document-intelligence.onrender.com) — free tier, so the first load after idle takes up to a minute while the instance wakes.

A domain-adaptive document intelligence system that combines NLP, LLMs, agentic workflows, and MLOps/LLMOps/AgentOps practices into one resume-grade project.

The platform is designed to ingest long documents, build searchable knowledge bases, answer questions with citations, compare documents, generate structured reports, and track the full lifecycle of data, prompts, models, evaluations, and deployments.

## Project Positioning

This is not a crypto-only project. The core system is domain-agnostic, with configurable domain presets.

Initial target domains:

- Academic papers
- Financial and crypto reports
- Legal or policy documents
- Technical documentation

## Core Capabilities

- Upload and parse PDFs, text files, and web documents.
- Clean, chunk, embed, and index document collections.
- Run RAG over document collections with cited answers.
- Use an agent graph for planning, retrieval, verification, and report writing.
- Extract NLP signals such as entities, topics, claims, risks, methods, limitations, and metrics.
- Evaluate retrieval quality, citation accuracy, answer relevance, hallucination risk, and latency.
- Track experiments, prompts, model versions, metrics, artifacts, and datasets.
- Monitor production inputs for text quality changes and drift.
- Serve local open-source LLMs on a 40 GB GPU using quantization where useful.
- Implement explicit MLOps, LLMOps, and AgentOps pipelines.

## Intended Resume Story

Built an agentic document intelligence platform using RAG, open-source LLMs, LangGraph-style agent orchestration, FastAPI, vector search, MLflow, DVC, Docker, and drift monitoring. Implemented document ingestion, semantic retrieval, multi-step reasoning agents, cited report generation, LoRA/QLoRA fine-tuning, model evaluation, and reproducible MLOps pipelines.

## Ops Pipelines

This project will include three connected operational pipelines:

- MLOps pipeline: data versioning, experiment tracking, model registry, reproducible training, deployment, and monitoring.
- LLMOps pipeline: prompt versioning, RAG evaluation, model serving, token/latency tracking, hallucination checks, and prompt/model comparisons.
- AgentOps pipeline: agent workflow tracing, tool-call logs, state transition tracking, planner/retriever/verifier observability, failure replay, and guardrail checks.

See [docs/OPS_PIPELINES.md](docs/OPS_PIPELINES.md) for the detailed design.

## Recommended First MVP

The first useful version should be small but real:

1. Ingest PDFs and text files.
2. Parse and chunk documents.
3. Generate embeddings and store them in a vector database.
4. Answer user questions with cited document chunks.
5. Generate a structured research brief.
6. Track ingestion, retrieval, and generation runs in MLflow.
7. Version sample datasets with DVC.
8. Add basic evaluations for retrieval and answer quality.

## Current First Step: Ingestion

The first implemented command parses supported documents and writes traceable JSONL chunks.

Run with the existing Conda environment:

```bash
conda run -n crypto_env python -m pytest
```

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.ingestion \
  --input data/raw \
  --output data/processed/chunks.jsonl \
  --chunk-size 800 \
  --chunk-overlap 120
```

Supported input types:

- `.txt`
- `.md`
- `.pdf` through the system `pdftotext` command

## Current Second Step: Baseline RAG

The second implemented layer builds a local retrieval index over the chunk JSONL file and returns cited evidence for a question.

Build an index:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.rag.index \
  --chunks data/processed/chunks.jsonl \
  --index data/processed/vector_index
```

Build a dense index:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.rag.index \
  --chunks data/processed/chunks.jsonl \
  --index data/processed/dense_index \
  --backend dense \
  --embedding-model lsa
```

Query the index:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.rag.query \
  --index data/processed/vector_index \
  --question "What does the platform do with documents?" \
  --top-k 3
```

Run the starter retrieval evaluation:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.rag.evaluate \
  --index data/processed/vector_index \
  --golden data/reference/golden_qa.jsonl \
  --top-k 3
```

The project includes a local `scikit-learn` TF-IDF vector baseline, dense retrieval, and a `hybrid` backend that fuses BM25 with dense rankings via weighted reciprocal-rank fusion (`--backend hybrid`, tunable with `--rrf-k` and `--hybrid-dense-weight`). Dense retrieval defaults to dependency-light LSA embeddings and can use `sentence-transformers` plus FAISS when those optional dependencies are installed. When a reranker is enabled, the first-stage candidate pool automatically widens to 3x `top_k` (at least 10) unless `--candidate-k` is set explicitly.

Compare retrieval backends:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.mlops.run_retrieval_benchmark \
  --chunks data/processed/chunks.jsonl \
  --golden data/reference/golden_qa.jsonl \
  --backends tfidf dense hybrid \
  --rerankers none lexical cross_encoder \
  --candidate-k 10 \
  --cross-encoder-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --cross-encoder-batch-size 8 \
  --allow-reranker-download \
  --top-k 3
```

## Current Third Step: Agentic Workflow

The third implemented layer wraps retrieval in an inspectable agent workflow:

```text
intent_router -> planner -> retriever -> evidence_verifier -> writer -> citation_checker
```

Run a cited Q&A agent:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.agents \
  --index data/processed/vector_index \
  --question "What does the platform do with documents?" \
  --task qa \
  --top-k 3
```

Run a research brief agent:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.agents \
  --index data/processed/vector_index \
  --question "Create a research brief about the document intelligence platform." \
  --task brief \
  --domain academic \
  --top-k 3
```

Each agent run writes an AgentOps trace JSON file under `data/monitoring/agent_traces/` unless `--no-trace` is passed.

## Current Fourth Step: MLOps Foundation

The fourth implemented layer adds reproducible tracked runs:

- Local MLOps run records under `data/monitoring/mlops_runs/`
- Tracked ingestion command
- Tracked RAG indexing and evaluation command
- Tracked agent smoke-test command
- `params.yaml`
- DVC-compatible `dvc.yaml`
- Docker smoke-test config
- GitHub Actions CI template
- Optional MLflow hooks

See [docs/MLOPS.md](docs/MLOPS.md) for commands and setup notes.

## Current Fifth Step: LLMOps Foundation

The fifth implemented layer adds prompt and generation observability:

- Versioned prompts in `prompts/`
- Prompt hashes
- Deterministic grounded generation baseline
- Optional Hugging Face text-generation adapter
- Token and latency metrics
- Citation coverage checks
- Unsupported sentence checks
- LLMOps reports
- Agent writer integration

Run a standalone LLMOps smoke test:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.llmops \
  --index data/processed/vector_index \
  --question "What does the platform do with documents?" \
  --task qa \
  --top-k 3
```

Run the tracked version:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.mlops.run_llmops_smoke \
  --index data/processed/vector_index \
  --question "What does the platform do with documents?" \
  --task qa \
  --top-k 3
```

See [docs/LLMOPS.md](docs/LLMOPS.md) for details.

## Answer-Quality Evaluation

Beyond retrieval metrics, the project scores generated answers for faithfulness (grounding in retrieved evidence), answer relevance, expected-fact coverage, and citation coverage over the golden Q&A set. It is deterministic by default (extractive baseline), so it runs in CI, and the same harness can drive any writer for model comparisons.

Run it (tracked as an MLOps run):

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.mlops.run_generation_eval \
  --index data/processed/vector_index \
  --golden data/reference/golden_qa.jsonl \
  --top-k 5
```

Compare a hosted writer (for example Gemini with thinking off):

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.mlops.run_generation_eval \
  --provider openai_compatible \
  --model-name gemini-2.5-flash \
  --endpoint-url https://generativelanguage.googleapis.com/v1beta/openai/chat/completions \
  --reasoning-effort none
```

An optional **LLM-as-judge** pass (`--judge-model-name ... --judge-endpoint-url ... --judge-api-key ...`) scores the same answers semantically and reports lexical-vs-judge agreement (mean gap + Pearson correlation), quantifying how far the cheap CI-safe proxy can be trusted. See [docs/LLMOPS.md](docs/LLMOPS.md#llm-as-judge-optional-second-opinion).

The latest report is surfaced as the dashboard's **Answer Quality** tiles and `GET /monitoring/generation-eval`. The CI quality gate runs this over a **corpus of real public documents** (`data/eval/` — GDPR/EU AI Act, IETF RFCs, NIST, SEC, arXiv, 18 docs across 5 categories, 45 golden questions) so the numbers are not overfit to project-authored text. On that corpus the extractive baseline scores about **0.60 faithfulness, 0.96 grounded rate, and 0.80 expected coverage** (retrieval is saturated at 1.0 because the domains are lexically distinct, so faithfulness is the discriminating metric). See [docs/EVALUATION_DATASET.md](docs/EVALUATION_DATASET.md).

## Continuous Integration

Every push and pull request runs a [GitHub Actions pipeline](.github/workflows/ci.yml) with two stages:

1. **Tests** — the full `pytest` suite across Python 3.10-3.14, on a minimal dependency install (no torch/faiss/transformers needed).
2. **Eval quality gate** — rebuilds the index, runs the deterministic retrieval and answer-quality evaluations, then fails the build if any metric falls outside its threshold in [`ci/eval_thresholds.json`](ci/eval_thresholds.json).

Because the baseline evals are deterministic, the gate is stable rather than flaky. Reproduce it locally:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.mlops.run_rag_eval \
  --backend tfidf --no-faiss --reranker none --top-k 5
conda run -n crypto_env env PYTHONPATH=src python -m adip.mlops.run_generation_eval
conda run -n crypto_env env PYTHONPATH=src python -m adip.mlops.eval_gate \
  --thresholds ci/eval_thresholds.json
```

See [docs/CICD.md](docs/CICD.md) for the gate design and how to update thresholds. The badge above reflects the latest run (update the `Muditkumar123/<repo>` slug if you name the repository differently).

## Deployment

The project ships as a single self-contained container (API + dashboard, demo index baked in, no GPU or model download). Run it in one command:

```bash
docker compose -f infra/docker/docker-compose.yml up --build
# open http://localhost:8010   (health: curl http://localhost:8010/health)
```

The image is a slim multi-stage build with pinned runtime deps, a non-root user, a `/health` healthcheck, and `$PORT` support for hosted platforms. A [`render.yaml`](render.yaml) blueprint gives a one-click Render deploy, and CI builds + smoke-tests the image on every push. Full guide: [docs/DEPLOY.md](docs/DEPLOY.md).

## Model Profiles

The project now has selectable model profiles in `config/model_profiles.yaml`:

- `extractive_baseline`
- `qwen3_8b_default`
- `deepseek_r1_distill_qwen_14b_reasoning`
- `deepseek_r1_distill_qwen_32b_stretch`
- `deepseek_v4_flash_cloud`
- `deepseek_v4_pro_cloud`
- `deepseek_v3_2_cloud_benchmark`

List them:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.config.list_model_profiles
```

Use one in LLMOps or agents:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.llmops \
  --index data/processed/vector_index \
  --question "What does the platform do with documents?" \
  --model-profile extractive_baseline
```

See [docs/MODEL_PROFILES.md](docs/MODEL_PROFILES.md) for Qwen and DeepSeek usage, and [docs/API_KEYS.md](docs/API_KEYS.md) for hosted model API key setup.

## Local LLM Serving

The serving layer can inspect GPU/package readiness, generate launch plans, run a local Transformers backend, and start a minimal OpenAI-compatible server.

Inspect:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.serving inspect
```

Generate a Qwen launch plan:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.serving launch-plan \
  --model-profile qwen3_8b_default
```

See [docs/SERVING.md](docs/SERVING.md) for serving details.

Qwen3-8B has been downloaded and verified locally. See [docs/QWEN3_LOCAL_SMOKE.md](docs/QWEN3_LOCAL_SMOKE.md).

DeepSeek-R1-Distill-Qwen-14B has also been downloaded and verified locally as a reasoning planner/verifier. See [docs/DEEPSEEK14B_LOCAL_SMOKE.md](docs/DEEPSEEK14B_LOCAL_SMOKE.md).

## Application API

The project now exposes the document workflow through a FastAPI service:

- `GET /health`
- `GET /monitoring/retrieval-benchmark`
- `GET /monitoring/generation-eval`
- `GET /model-profiles`
- `GET /history/agent-traces`
- `GET /history/mlops-runs`
- `GET /documents`
- `POST /documents/upload`
- `DELETE /documents/{filename}`
- `POST /pipeline/rebuild-index`
- `POST /rag/query`
- `POST /agent/run`

Start it:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.api \
  --host 127.0.0.1 \
  --port 8010
```

Then open:

```text
http://127.0.0.1:8010/
```

OpenAPI docs are available at `http://127.0.0.1:8010/docs`.

See [docs/API.md](docs/API.md) for example requests and [docs/DOCKER_DEMO.md](docs/DOCKER_DEMO.md) for the containerized dashboard demo.

Recommended dashboard setup for readable local LLM answers:

- Writer model: `qwen3_8b_default`
- Reasoning model: `deepseek_r1_distill_qwen_14b_reasoning`
- Device: `CUDA 1`

## Project Notes

- [docs/EVALUATION_DATASET.md](docs/EVALUATION_DATASET.md): current sample corpus, golden QA set, and benchmark meaning.
- [docs/FIXES.md](docs/FIXES.md): problems found in the LLM answer path and how they were fixed, with regression tests.

## Planned Stack

- Backend: Python, FastAPI
- Agent orchestration: LangGraph-style graph workflow
- NLP/LLM: Hugging Face Transformers, Sentence Transformers, PEFT
- RAG storage: FAISS or Chroma
- Baseline retrieval: scikit-learn TF-IDF local vector index
- Serving: vLLM or Transformers-based local inference
- MLOps: MLflow, DVC, Docker, GitHub Actions
- Monitoring: Evidently or custom text drift reports
- UI: Streamlit first, optional Next.js later

## Repository Layout

```text
Agentic Document Intelligence Platform/
  docs/
    API.md
    API_KEYS.md
    CICD.md
    DEEPSEEK14B_LOCAL_SMOKE.md
    DESIGN.md
    DOCKER_DEMO.md
    EVALUATION_DATASET.md
    FIXES.md
    LLMOPS.md
    MODEL_PROFILES.md
    MLOPS.md
    OPS_PIPELINES.md
    QWEN3_LOCAL_SMOKE.md
    ROADMAP.md
    SERVING.md
  src/adip/
    agents/
    api/
    config/
    evaluation/
    ingestion/
    llmops/
    rag/
    schemas/
    serving/
  data/
    raw/
    processed/
    reference/
    monitoring/
  models/
  notebooks/
  tests/
  ci/
    eval_thresholds.json
  .github/
    workflows/
      ci.yml
  infra/
    docker/
```

## Design Principle

The project should prove practical AI engineering, not just chatbot construction. Every LLM feature should have an evaluation, every model/data artifact should be traceable, and every agent output should be grounded in document evidence.
