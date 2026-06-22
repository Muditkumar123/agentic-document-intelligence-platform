"""Starter retrieval evaluation CLI."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

from adip.rag.rerank import DEFAULT_CROSS_ENCODER_MODEL, rerank_results
from adip.rag.retriever import load_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adip.rag.evaluate",
        description="Evaluate retrieval against a small golden Q&A JSONL file.",
    )
    parser.add_argument(
        "--index",
        "-x",
        type=Path,
        default=Path("data/processed/vector_index"),
        help="Directory containing the saved RAG index.",
    )
    parser.add_argument(
        "--golden",
        "-g",
        type=Path,
        default=Path("data/reference/golden_qa.jsonl"),
        help="Golden questions JSONL path.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Number of first-stage candidates to retrieve before reranking.",
    )
    parser.add_argument(
        "--reranker",
        choices=["none", "lexical", "cross_encoder"],
        default="none",
        help="Optional second-stage reranker.",
    )
    parser.add_argument(
        "--rerank-weight",
        type=float,
        default=0.25,
        help="Weight for the original retrieval score in lexical reranking.",
    )
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CROSS_ENCODER_MODEL)
    parser.add_argument("--cross-encoder-device", default=None)
    parser.add_argument("--cross-encoder-batch-size", type=int, default=16)
    parser.add_argument(
        "--allow-reranker-download",
        action="store_true",
        help="Allow the cross-encoder reranker model to be downloaded if not cached.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional path for detailed evaluation JSON.",
    )
    return parser


def read_golden(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.expanduser().open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "question" not in row:
                raise ValueError(f"{path}:{line_number} is missing `question`")
            rows.append(row)
    if not rows:
        raise ValueError(f"No golden questions found in {path}")
    return rows


def evaluate(
    index_path: Path,
    golden_path: Path,
    top_k: int,
    candidate_k: int | None = None,
    reranker: str = "none",
    rerank_weight: float = 0.25,
    cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
    cross_encoder_device: str | None = None,
    cross_encoder_batch_size: int = 16,
    cross_encoder_local_files_only: bool = True,
) -> dict[str, Any]:
    index = load_index(index_path)
    golden = read_golden(golden_path)
    results: list[dict[str, Any]] = []
    hits = 0
    reciprocal_ranks: list[float] = []
    query_latencies: list[float] = []
    start_time = time.perf_counter()
    resolved_candidate_k = candidate_k or top_k
    if resolved_candidate_k < top_k:
        raise ValueError("candidate_k must be greater than or equal to top_k")

    for row in golden:
        query_start = time.perf_counter()
        candidates = index.search(row["question"], top_k=resolved_candidate_k)
        retrieved = rerank_results(
            row["question"],
            candidates,
            reranker=reranker,
            top_k=top_k,
            original_score_weight=rerank_weight,
            cross_encoder_model=cross_encoder_model,
            cross_encoder_device=cross_encoder_device,
            cross_encoder_batch_size=cross_encoder_batch_size,
            cross_encoder_local_files_only=cross_encoder_local_files_only,
        )
        query_latencies.append((time.perf_counter() - query_start) * 1000)
        expected_chunk_ids = set(row.get("expected_chunk_ids", []))
        expected_substrings = [text.lower() for text in row.get("expected_substrings", [])]

        rank = find_hit_rank(retrieved, expected_chunk_ids, expected_substrings)
        hit = rank is not None
        if hit:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        results.append(
            {
                "question": row["question"],
                "hit": hit,
                "rank": rank,
                "retrieved_chunk_ids": [item.chunk["chunk_id"] for item in retrieved],
                "retrieved_citations": [item.citation_label for item in retrieved],
            }
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    total = len(golden)
    return {
        "backend": index.backend,
        "chunk_count": len(index.chunks),
        "embedding_model": index.embedding_model,
        "index_path": str(index_path),
        "index_size_bytes": path_size_bytes(index_path),
        "index_metadata": index.metadata,
        "index_vocabulary_size": index.vocabulary_size,
        "golden_path": str(golden_path),
        "top_k": top_k,
        "candidate_k": resolved_candidate_k,
        "reranker": reranker,
        "rerank_weight": rerank_weight,
        "cross_encoder_model": cross_encoder_model if reranker == "cross_encoder" else None,
        "cross_encoder_batch_size": cross_encoder_batch_size if reranker == "cross_encoder" else None,
        "question_count": total,
        "hit_rate_at_k": hits / total,
        "mrr": sum(reciprocal_ranks) / total,
        "elapsed_ms": elapsed_ms,
        "avg_query_latency_ms": sum(query_latencies) / total,
        "results": results,
    }


def path_size_bytes(path: Path) -> int:
    expanded = path.expanduser()
    if expanded.is_file():
        return expanded.stat().st_size
    if not expanded.exists():
        return 0
    return sum(item.stat().st_size for item in expanded.rglob("*") if item.is_file())


def find_hit_rank(
    retrieved,
    expected_chunk_ids: set[str],
    expected_substrings: list[str],
) -> int | None:
    for rank, item in enumerate(retrieved, start=1):
        chunk_id = item.chunk["chunk_id"]
        text = item.chunk["text"].lower()
        if chunk_id in expected_chunk_ids:
            return rank
        if any(expected in text for expected in expected_substrings):
            return rank
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
