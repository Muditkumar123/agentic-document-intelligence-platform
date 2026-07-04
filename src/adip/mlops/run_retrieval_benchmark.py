"""Tracked retrieval backend benchmark command."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

from adip.mlops.tracking import start_run
from adip.rag.chunks import read_chunks_jsonl
from adip.rag.evaluate import evaluate
from adip.rag.rerank import DEFAULT_CROSS_ENCODER_MODEL
from adip.rag.retriever import build_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adip.mlops.run_retrieval_benchmark",
        description="Compare retrieval backends and log an MLOps benchmark record.",
    )
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--index-root", type=Path, default=Path("data/processed/retrieval_benchmark"))
    parser.add_argument("--golden", type=Path, default=Path("data/reference/golden_qa.jsonl"))
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["tfidf", "dense", "dense_lsa", "sentence_transformers", "hybrid"],
        default=["tfidf", "dense", "hybrid"],
    )
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--embedding-model", default="lsa")
    parser.add_argument("--dense-dimensions", type=int, default=128)
    parser.add_argument("--no-faiss", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="First-stage candidate count before reranking. Defaults to max(top_k, top_k * 3).",
    )
    parser.add_argument(
        "--rerankers",
        nargs="+",
        choices=["none", "lexical", "cross_encoder"],
        default=["none", "lexical"],
        help="Second-stage rerankers to evaluate for each backend.",
    )
    parser.add_argument("--rerank-weight", type=float, default=0.25)
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CROSS_ENCODER_MODEL)
    parser.add_argument("--cross-encoder-device", default=None)
    parser.add_argument("--cross-encoder-batch-size", type=int, default=16)
    parser.add_argument("--allow-reranker-download", action="store_true")
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("data/monitoring/retrieval_benchmark_report.json"),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("data/monitoring/retrieval_benchmark_metrics.json"),
    )
    parser.add_argument("--run-dir", type=Path, default=Path("data/monitoring/mlops_runs"))
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with start_run(
        "retrieval_backend_benchmark",
        run_dir=args.run_dir,
        enable_mlflow=args.enable_mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        tags={"pipeline": "rag", "benchmark": "retrieval_backends"},
    ) as run:
        run.log_params(
            {
                "chunks": str(args.chunks),
                "index_root": str(args.index_root),
                "golden": str(args.golden),
                "backends": ",".join(args.backends),
                "ngram_max": args.ngram_max,
                "embedding_model": args.embedding_model,
                "dense_dimensions": args.dense_dimensions,
                "faiss_requested": not args.no_faiss,
                "top_k": args.top_k,
                "candidate_k": args.candidate_k or max(args.top_k, args.top_k * 3),
                "rerankers": ",".join(args.rerankers),
                "rerank_weight": args.rerank_weight,
                "cross_encoder_model": args.cross_encoder_model,
                "cross_encoder_device": args.cross_encoder_device or "",
                "cross_encoder_batch_size": args.cross_encoder_batch_size,
                "cross_encoder_local_files_only": not args.allow_reranker_download,
            }
        )
        chunks = read_chunks_jsonl(args.chunks)
        report = run_benchmark(
            chunks=chunks,
            index_root=args.index_root,
            golden_path=args.golden,
            backends=args.backends,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            rerankers=args.rerankers,
            rerank_weight=args.rerank_weight,
            cross_encoder_model=args.cross_encoder_model,
            cross_encoder_device=args.cross_encoder_device,
            cross_encoder_batch_size=args.cross_encoder_batch_size,
            cross_encoder_local_files_only=not args.allow_reranker_download,
            ngram_max=args.ngram_max,
            embedding_model=args.embedding_model,
            dense_dimensions=args.dense_dimensions,
            use_faiss=not args.no_faiss,
        )
        metrics = benchmark_metrics(report)
        run.log_metrics(metrics)
        run.log_params(
            {
                "best_backend_by_mrr": report["best_backend_by_mrr"],
                "best_variant_by_mrr": report["best_variant_by_mrr"],
            }
        )

        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        run.log_artifact("benchmark_report", args.report_output)
        run.log_artifact("benchmark_metrics", args.metrics_output)
        run.log_artifact("index_root", args.index_root)

    payload = {
        "best_backend_by_mrr": report["best_backend_by_mrr"],
        "best_variant_by_mrr": report["best_variant_by_mrr"],
        "metrics": metrics,
        "metrics_output": str(args.metrics_output),
        "mlops_run_path": str(run.run_path),
        "report_output": str(args.report_output),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_benchmark(
    chunks: list[dict[str, Any]],
    index_root: Path,
    golden_path: Path,
    backends: list[str],
    top_k: int,
    candidate_k: int | None,
    rerankers: list[str],
    rerank_weight: float,
    cross_encoder_model: str,
    cross_encoder_device: str | None,
    cross_encoder_batch_size: int,
    cross_encoder_local_files_only: bool,
    ngram_max: int,
    embedding_model: str,
    dense_dimensions: int,
    use_faiss: bool,
) -> dict[str, Any]:
    root = index_root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    backend_reports: dict[str, Any] = {}
    variant_reports: dict[str, Any] = {}
    resolved_candidate_k = candidate_k or max(top_k, top_k * 3)

    for backend in backends:
        index_path = root / backend
        if index_path.exists():
            shutil.rmtree(index_path)

        build_start = time.perf_counter()
        index = build_index(
            chunks,
            backend=backend,
            ngram_max=ngram_max,
            embedding_model=embedding_model,
            dense_dimensions=dense_dimensions,
            use_faiss=use_faiss,
        )
        index.save(index_path)
        build_elapsed_ms = (time.perf_counter() - build_start) * 1000

        for reranker in rerankers:
            evaluation = evaluate(
                index_path,
                golden_path,
                top_k=top_k,
                candidate_k=resolved_candidate_k,
                reranker=reranker,
                rerank_weight=rerank_weight,
                cross_encoder_model=cross_encoder_model,
                cross_encoder_device=cross_encoder_device,
                cross_encoder_batch_size=cross_encoder_batch_size,
                cross_encoder_local_files_only=cross_encoder_local_files_only,
            )
            evaluation["build_elapsed_ms"] = build_elapsed_ms
            variant_name = retrieval_variant_name(index.backend, reranker)
            variant_reports[variant_name] = evaluation
            if reranker == "none":
                backend_reports[index.backend] = evaluation

    best_backend = best_report_by_mrr(backend_reports)
    best_variant = best_report_by_mrr(variant_reports)
    return {
        "backend_count": len(backend_reports),
        "variant_count": len(variant_reports),
        "best_backend_by_mrr": best_backend,
        "best_variant_by_mrr": best_variant,
        "chunk_count": len(chunks),
        "candidate_k": resolved_candidate_k,
        "golden_path": str(golden_path),
        "rerankers": rerankers,
        "rerank_weight": rerank_weight,
        "cross_encoder_model": cross_encoder_model if "cross_encoder" in rerankers else None,
        "cross_encoder_batch_size": cross_encoder_batch_size if "cross_encoder" in rerankers else None,
        "top_k": top_k,
        "backends": backend_reports,
        "variants": variant_reports,
    }


def benchmark_metrics(report: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {
        "backend_count": float(report["backend_count"]),
        "variant_count": float(report["variant_count"]),
        "chunk_count": float(report["chunk_count"]),
    }
    for variant, variant_report in report["variants"].items():
        prefix = sanitize_metric_prefix(variant)
        metrics[f"{prefix}_hit_rate_at_k"] = float(variant_report["hit_rate_at_k"])
        metrics[f"{prefix}_mrr"] = float(variant_report["mrr"])
        metrics[f"{prefix}_avg_query_latency_ms"] = float(variant_report["avg_query_latency_ms"])
        metrics[f"{prefix}_build_elapsed_ms"] = float(variant_report["build_elapsed_ms"])
        metrics[f"{prefix}_index_size_bytes"] = float(variant_report["index_size_bytes"])
        metrics[f"{prefix}_faiss_enabled"] = 1.0 if variant_report["index_metadata"].get("faiss_enabled") else 0.0

    if "tfidf" in report["backends"]:
        tfidf = report["backends"]["tfidf"]
        for backend, backend_report in report["backends"].items():
            if backend == "tfidf":
                continue
            prefix = sanitize_metric_prefix(backend)
            metrics[f"{prefix}_minus_tfidf_hit_rate_at_k"] = float(
                backend_report["hit_rate_at_k"] - tfidf["hit_rate_at_k"]
            )
            metrics[f"{prefix}_minus_tfidf_mrr"] = float(backend_report["mrr"] - tfidf["mrr"])
    for variant, variant_report in report["variants"].items():
        reranker = variant_report.get("reranker")
        backend = variant_report.get("backend")
        if reranker == "none":
            continue
        baseline = report["backends"].get(backend)
        if baseline is None:
            continue
        prefix = sanitize_metric_prefix(variant)
        metrics[f"{prefix}_minus_plain_hit_rate_at_k"] = float(
            variant_report["hit_rate_at_k"] - baseline["hit_rate_at_k"]
        )
        metrics[f"{prefix}_minus_plain_mrr"] = float(variant_report["mrr"] - baseline["mrr"])
    return metrics


def best_report_by_mrr(reports: dict[str, Any]) -> str:
    if not reports:
        raise ValueError("At least one report is required")
    return max(
        reports.items(),
        key=lambda item: (item[1]["mrr"], item[1]["hit_rate_at_k"]),
    )[0]


def sanitize_metric_prefix(value: str) -> str:
    return value.replace("-", "_").replace("/", "_")


def retrieval_variant_name(backend: str, reranker: str) -> str:
    if reranker == "none":
        return backend
    return f"{backend}_{reranker}_rerank"


if __name__ == "__main__":
    raise SystemExit(main())
