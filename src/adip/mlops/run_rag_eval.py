"""Tracked RAG indexing and evaluation command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from adip.rag.chunks import read_chunks_jsonl
from adip.rag.evaluate import evaluate
from adip.rag.rerank import DEFAULT_CROSS_ENCODER_MODEL
from adip.rag.retriever import build_index
from adip.mlops.tracking import start_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adip.mlops.run_rag_eval",
        description="Build a RAG index, evaluate retrieval, and log an MLOps run record.",
    )
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--index", type=Path, default=Path("data/processed/vector_index"))
    parser.add_argument("--golden", type=Path, default=Path("data/reference/golden_qa.jsonl"))
    parser.add_argument("--backend", choices=["tfidf", "dense", "dense_lsa", "sentence_transformers"], default="tfidf")
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--embedding-model", default="lsa")
    parser.add_argument("--dense-dimensions", type=int, default=128)
    parser.add_argument("--no-faiss", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--reranker", choices=["none", "lexical", "cross_encoder"], default="none")
    parser.add_argument("--rerank-weight", type=float, default=0.25)
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CROSS_ENCODER_MODEL)
    parser.add_argument("--cross-encoder-device", default=None)
    parser.add_argument("--cross-encoder-batch-size", type=int, default=16)
    parser.add_argument("--allow-reranker-download", action="store_true")
    parser.add_argument("--metrics-output", type=Path, default=Path("data/monitoring/rag_eval_metrics.json"))
    parser.add_argument("--run-dir", type=Path, default=Path("data/monitoring/mlops_runs"))
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with start_run(
        "rag_index_eval",
        run_dir=args.run_dir,
        enable_mlflow=args.enable_mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        tags={"pipeline": "rag", "backend": args.backend},
    ) as run:
        run.log_params(
            {
                "chunks": str(args.chunks),
                "index": str(args.index),
                "golden": str(args.golden),
                "backend": args.backend,
                "ngram_max": args.ngram_max,
                "embedding_model": args.embedding_model,
                "dense_dimensions": args.dense_dimensions,
                "faiss_requested": not args.no_faiss,
                "top_k": args.top_k,
                "candidate_k": args.candidate_k or args.top_k,
                "reranker": args.reranker,
                "rerank_weight": args.rerank_weight,
                "cross_encoder_model": args.cross_encoder_model,
                "cross_encoder_device": args.cross_encoder_device or "",
                "cross_encoder_batch_size": args.cross_encoder_batch_size,
                "cross_encoder_local_files_only": not args.allow_reranker_download,
            }
        )
        chunks = read_chunks_jsonl(args.chunks)
        index = build_index(
            chunks,
            backend=args.backend,
            ngram_max=args.ngram_max,
            embedding_model=args.embedding_model,
            dense_dimensions=args.dense_dimensions,
            use_faiss=not args.no_faiss,
        )
        index.save(args.index)
        report = evaluate(
            args.index,
            args.golden,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            reranker=args.reranker,
            rerank_weight=args.rerank_weight,
            cross_encoder_model=args.cross_encoder_model,
            cross_encoder_device=args.cross_encoder_device,
            cross_encoder_batch_size=args.cross_encoder_batch_size,
            cross_encoder_local_files_only=not args.allow_reranker_download,
        )
        metrics = {
            "chunk_count": len(chunks),
            "vocabulary_size": index.vocabulary_size,
            "index_size_bytes": report["index_size_bytes"],
            "question_count": report["question_count"],
            "hit_rate_at_k": report["hit_rate_at_k"],
            "mrr": report["mrr"],
            "eval_elapsed_ms": report["elapsed_ms"],
            "avg_query_latency_ms": report["avg_query_latency_ms"],
            "faiss_enabled": 1.0 if index.metadata.get("faiss_enabled") else 0.0,
            "reranker_enabled": 0.0 if args.reranker == "none" else 1.0,
        }
        run.log_metrics(metrics)
        run.log_artifact("index_dir", args.index)
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        run.log_artifact("evaluation_report", args.metrics_output)

    payload = {
        "metrics": metrics,
        "metrics_output": str(args.metrics_output),
        "mlops_run_path": str(run.run_path),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
