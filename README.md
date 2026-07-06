# Agentic Document Intelligence Platform

[![CI](https://github.com/Muditkumar123/agentic-document-intelligence-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/Muditkumar123/agentic-document-intelligence-platform/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Live_Demo-onrender.com-2dd4bf?logo=render&logoColor=white)](https://agentic-document-intelligence.onrender.com)

A document Q&A system built around one rule: no LLM feature ships without an
evaluation for it. Upload documents, ask questions, get answers with citations
into the source text. An agent pipeline handles planning, retrieval, evidence
verification and writing, and a CI quality gate fails the build when the
numbers drop.

Live demo: [agentic-document-intelligence.onrender.com](https://agentic-document-intelligence.onrender.com)
(free tier, so the first load after idle takes about a minute).

## Features

- **Ingestion** for PDF, markdown and text with traceable chunking. An
  optional table-aware parser (unstructured.io) serializes table rows with
  their column headers so cells stay retrievable across chunk boundaries.
- **Retrieval**: TF-IDF, dense (LSA by default, sentence-transformers +
  FAISS optional) and hybrid (BM25 fused with dense rankings via
  reciprocal-rank fusion), plus optional cross-encoder reranking and
  multi-query rewriting.
- **Agent pipeline** on LangGraph: intent_router, planner, retriever,
  evidence_verifier, writer, citation_checker. Every run writes a trace that
  can be replayed in the dashboard. Falls back to a plain sequential engine
  when langgraph isn't installed.
- **Abstention**: when the retrieved evidence is too weak, the system says so
  instead of answering. Tested against unanswerable probe questions.
- **Writers**: a deterministic extractive baseline (runs anywhere, keeps CI
  stable), local Qwen3-8B or DeepSeek-R1-Distill through an OpenAI-compatible
  server, or any hosted endpoint.
- **Ops**: tracked runs, a DVC pipeline, a Docker image built and smoke-tested
  in CI, in-process index/query caching (~30x on repeated queries), and query
  drift monitoring (OOV rate, question-length shift, PSI against a
  golden-question baseline).

The heavy pieces are optional extras (`[api]`, `[agents]`, `[ragas]`,
`[tables]`, `[finetune]`), so the core installs and tests without torch,
transformers or faiss.

## Evaluation

The eval corpus is 19 real public documents (GDPR, EU AI Act, IETF RFCs,
NIST, SEC filings, arXiv papers) with 47 golden questions plus 10
unanswerable probes, so the numbers aren't overfit to project-authored text.

- A deterministic quality gate runs in CI: retrieval and answer metrics are
  checked against [`ci/eval_thresholds.json`](ci/eval_thresholds.json) on
  every push.
- On a paraphrase probe set (reworded questions with no keyword overlap),
  plain TF-IDF gets 0.85 hit@5; hybrid retrieval plus LLM query rewriting
  reaches 1.00 hit@5 / 0.90 MRR.
- LLM-as-judge and RAGAS passes score the same answers semantically. One
  finding worth reading: the same 8B model that scores 0.40 as a free-form
  faithfulness judge scores 0.925 through RAGAS claim decomposition
  ([docs/LLMOPS.md](docs/LLMOPS.md)).
- A LoRA fine-tuning experiment (chunk-category classifier with document-level
  splits) beats frozen-head and TF-IDF baselines while training under 1% of
  parameters ([docs/FINETUNING.md](docs/FINETUNING.md)).

## Quickstart

With Docker (demo index baked in, no GPU or model downloads):

```bash
docker compose -f infra/docker/docker-compose.yml up --build
# open http://localhost:8010
```

From source:

```bash
pip install -e ".[dev,api]"
python -m adip.mlops.run_ingestion --input data/eval/raw --output data/processed/chunks.jsonl
python -m adip.rag.index --chunks data/processed/chunks.jsonl --index data/processed/vector_index
python -m adip.api --host 127.0.0.1 --port 8010
```

The dashboard is at `http://127.0.0.1:8010/` and the OpenAPI docs at `/docs`.
Run the test suite with `pytest` (202 tests, hermetic, no GPU needed).

## API

The FastAPI service exposes the whole workflow: `POST /rag/query` (with
backend selection and caching), `POST /agent/run`, document upload/delete and
index rebuild, history endpoints for agent traces and tracked runs, and
monitoring endpoints for generation eval, retrieval benchmarks, drift and
cache stats. Examples in [docs/API.md](docs/API.md).

## Documentation

| Doc | What's in it |
| --- | --- |
| [docs/DESIGN.md](docs/DESIGN.md) | Architecture and component walkthrough |
| [docs/EVALUATION_DATASET.md](docs/EVALUATION_DATASET.md) | Corpus, golden QA set, paraphrase probes |
| [docs/LLMOPS.md](docs/LLMOPS.md) | Generation eval, LLM-as-judge, RAGAS |
| [docs/MLOPS.md](docs/MLOPS.md) | Tracked runs, DVC, MLflow hooks |
| [docs/CICD.md](docs/CICD.md) | CI stages and the eval quality gate |
| [docs/FINETUNING.md](docs/FINETUNING.md) | LoRA experiment design and results |
| [docs/SERVING.md](docs/SERVING.md) | Local model serving (Qwen, DeepSeek) |
| [docs/DEPLOY.md](docs/DEPLOY.md) | Container image and Render deployment |
| [docs/API.md](docs/API.md) | Endpoint reference with examples |
| [docs/ROADMAP.md](docs/ROADMAP.md) | What was built when, and what's next |
