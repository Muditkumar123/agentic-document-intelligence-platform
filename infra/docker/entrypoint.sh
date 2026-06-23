#!/bin/sh
# Container entrypoint: ensure a usable index exists (rebuild it if a mounted volume
# shadows the one baked into the image), then serve the API on $PORT (deploy platforms
# like Render/Fly inject their own PORT).
set -e

if [ ! -d data/processed/vector_index ]; then
  echo "[entrypoint] no index found, building from data/eval/raw..."
  python -m adip.mlops.run_ingestion --input data/eval/raw \
    --output data/processed/chunks.jsonl \
    --metrics-output data/monitoring/ingestion_metrics.json
  python -m adip.mlops.run_rag_eval --chunks data/processed/chunks.jsonl \
    --index data/processed/vector_index --golden data/eval/golden_qa.jsonl \
    --backend tfidf --no-faiss --metrics-output data/monitoring/rag_eval_metrics.json
  python -m adip.mlops.run_generation_eval --index data/processed/vector_index \
    --golden data/eval/golden_qa.jsonl --abstention-threshold 0.10 \
    --metrics-output data/monitoring/generation_eval_metrics.json \
    --report-output data/monitoring/generation_eval_report.json
fi

exec python -m adip.api --host 0.0.0.0 --port "${PORT:-8010}"
