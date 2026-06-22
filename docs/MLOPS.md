# MLOps Foundation

This project has a working local MLOps tracker now and optional integration points for MLflow and DVC.

## What Is Implemented

- Local run records under `data/monitoring/mlops_runs/`.
- Tracked ingestion command.
- Tracked RAG indexing and retrieval evaluation command.
- Tracked retrieval backend benchmark command.
- Tracked agent smoke-test command.
- DVC-compatible `dvc.yaml`.
- Parameter file: `params.yaml`.
- Docker smoke-test config.
- GitHub Actions CI template.
- Optional MLflow logging when `mlflow` is installed.

## Run The Tracked Pipeline Commands

Ingestion:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.mlops.run_ingestion \
  --input data/raw \
  --output data/processed/chunks.jsonl \
  --chunk-size 800 \
  --chunk-overlap 120
```

RAG indexing and evaluation:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.mlops.run_rag_eval \
  --chunks data/processed/chunks.jsonl \
  --index data/processed/vector_index \
  --golden data/reference/golden_qa.jsonl \
  --top-k 3
```

Retrieval backend benchmark:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.mlops.run_retrieval_benchmark \
  --chunks data/processed/chunks.jsonl \
  --index-root data/processed/retrieval_benchmark \
  --golden data/reference/golden_qa.jsonl \
  --backends tfidf dense \
  --rerankers none lexical cross_encoder \
  --embedding-model lsa \
  --candidate-k 10 \
  --cross-encoder-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --cross-encoder-batch-size 8 \
  --allow-reranker-download \
  --top-k 3
```

This builds and evaluates multiple retrievers on the same golden questions, then logs metrics such as `tfidf_mrr`, `dense_lsa_mrr`, `tfidf_lexical_rerank_mrr`, `dense_lsa_lexical_rerank_mrr`, `tfidf_cross_encoder_rerank_mrr`, reranker deltas, query latency, index size, and whether FAISS was used.

Latest local benchmark result: `dense_lsa` is the best plain backend by MRR, while `tfidf_cross_encoder_rerank` is the best overall variant by MRR. Plain TF-IDF reached about 0.733 MRR, Dense LSA reached about 0.756 MRR, and both cross-encoder reranked variants reached about 0.900 MRR on the current 15-question golden set.

Agent smoke test:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.mlops.run_agent \
  --index data/processed/vector_index \
  --question "What does the platform do with documents?" \
  --task qa \
  --top-k 3
```

LLMOps smoke test:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.mlops.run_llmops_smoke \
  --index data/processed/vector_index \
  --question "What does the platform do with documents?" \
  --task qa \
  --top-k 3
```

## MLflow

`crypto_env` does not currently include MLflow. The tracker automatically uses MLflow when it is installed and `--enable-mlflow` is passed.

Example after installing MLflow:

```bash
conda run -n crypto_env env PYTHONPATH=src python -m adip.mlops.run_rag_eval \
  --enable-mlflow \
  --mlflow-tracking-uri file:./mlruns
```

## DVC

`crypto_env` does not currently include DVC. The project includes `dvc.yaml` and `params.yaml`, so after installing DVC the pipeline can be initialized with:

```bash
dvc init
dvc repro
dvc metrics show
```

The stages are:

- `ingest`
- `rag_eval`
- `retrieval_benchmark`
- `llmops_smoke`
- `agent_smoke`

## Docker

Build and run the smoke-test image from the project root:

```bash
docker build -f infra/docker/Dockerfile -t adip-smoke .
docker run --rm adip-smoke
```

Or use compose:

```bash
docker compose -f infra/docker/docker-compose.yml up --build
```
