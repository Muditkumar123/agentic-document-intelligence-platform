"""Benchmark the cross-encoder reranker on PyTorch vs ONNX Runtime (FP32 and INT8) on CPU.

Workload: every answerable golden and paraphrase question retrieves ``--candidates``
chunks with TF-IDF over the evaluation corpus, and each runtime scores all of them.
Reported per runtime and thread count: latency per question (p50/p95), pairs per
second, score difference from PyTorch, agreement of the reranked top-k with
PyTorch's, and hit@k / MRR@k against the expected answer substrings.

    python -m adip.mlops.run_reranker_benchmark --model <name-or-local-path> --onnx-dir data/models/onnx-minilm
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from adip.ingestion.pipeline import ingest_path
from adip.rag.chunks import read_chunks_jsonl
from adip.rag.evaluate import find_hit_rank, read_golden
from adip.rag.onnx_reranker import FP32_FILENAME, INT8_FILENAME, export_cross_encoder, score_onnx_pairs
from adip.rag.rerank import DEFAULT_CROSS_ENCODER_MODEL, score_cross_encoder_pairs
from adip.rag.retriever import RetrievedChunk, build_index

DEFAULT_CORPORA = (Path("data/eval/raw"), Path("data/eval/long"))
DEFAULT_QUESTION_FILES = (Path("data/eval/golden_qa.jsonl"), Path("data/eval/paraphrase_probes.jsonl"))
WARMUP_QUESTIONS = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_CROSS_ENCODER_MODEL, help="Hugging Face id or local model folder")
    parser.add_argument("--onnx-dir", type=Path, default=Path("data/models/onnx-reranker"))
    parser.add_argument("--threads", default="1,4", help="comma-separated CPU thread counts to test")
    parser.add_argument("--candidates", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("data/monitoring/reranker_benchmark.json"))
    return parser


def build_workload(candidates: int) -> list[dict[str, Any]]:
    """(question, candidate chunks, expected substrings) for every answerable question."""
    with tempfile.TemporaryDirectory() as tmp:
        chunks: list[dict[str, Any]] = []
        for number, corpus in enumerate(DEFAULT_CORPORA):
            output = Path(tmp) / f"chunks_{number}.jsonl"
            ingest_path(corpus, output)
            chunks.extend(read_chunks_jsonl(output))
    index = build_index(chunks, backend="tfidf", use_faiss=False)
    workload = []
    for path in DEFAULT_QUESTION_FILES:
        for row in read_golden(path):
            if row.get("answerable", True):
                workload.append(
                    {
                        "question": row["question"],
                        "candidates": index.search(row["question"], top_k=candidates),
                        "expected": [" ".join(text.lower().split()) for text in row.get("expected_substrings", [])],
                    }
                )
    return workload


def run_runtime(
    workload: list[dict[str, Any]], score: Callable[[str, list[str]], list[float]], top_k: int
) -> dict[str, Any]:
    for item in workload[:WARMUP_QUESTIONS]:
        score(item["question"], [chunk.chunk["text"] for chunk in item["candidates"]])
    latencies, all_scores, rankings, hits, reciprocal = [], [], [], 0, []
    for item in workload:
        texts = [chunk.chunk["text"] for chunk in item["candidates"]]
        start = time.perf_counter()
        scores = score(item["question"], texts)
        latencies.append((time.perf_counter() - start) * 1000)
        order = sorted(range(len(scores)), key=lambda position: -scores[position])[:top_k]
        reranked = [
            RetrievedChunk(chunk=item["candidates"][position].chunk, score=scores[position], rank=rank)
            for rank, position in enumerate(order, start=1)
        ]
        rank = find_hit_rank(reranked, set(), item["expected"])
        hits += rank is not None
        reciprocal.append(1.0 / rank if rank else 0.0)
        all_scores.append(scores)
        rankings.append(order)
    pairs = sum(len(item["candidates"]) for item in workload)
    ordered = sorted(latencies)
    return {
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))], 2),
        "pairs_per_second": round(pairs / (sum(latencies) / 1000), 1),
        f"hit_at_{top_k}": round(hits / len(workload), 4),
        f"mrr_at_{top_k}": round(sum(reciprocal) / len(workload), 4),
        "_scores": all_scores,
        "_rankings": rankings,
    }


def compare_to(reference: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    differences = [
        abs(a - b) for ref, oth in zip(reference["_scores"], other["_scores"]) for a, b in zip(ref, oth)
    ]
    pairs = list(zip(reference["_rankings"], other["_rankings"]))
    return {
        "max_abs_score_diff": float(f"{max(differences):.3g}"),
        "top1_agreement": round(sum(a[0] == b[0] for a, b in pairs) / len(pairs), 4),
        "topk_identical_order": round(sum(a == b for a, b in pairs) / len(pairs), 4),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    import torch

    torch_threads_before = torch.get_num_threads()
    if not (args.onnx_dir / FP32_FILENAME).is_file():
        export_cross_encoder(args.model, args.onnx_dir)
    workload = build_workload(args.candidates)
    sizes = {name: round((args.onnx_dir / name).stat().st_size / 1e6, 1) for name in (FP32_FILENAME, INT8_FILENAME)}
    report: dict[str, Any] = {
        "model": args.model,
        "questions": len(workload),
        "candidates_per_question": args.candidates,
        "top_k": args.top_k,
        "onnx_model_mb": sizes,
        "runs": {},
    }
    for threads in [int(value) for value in args.threads.split(",")]:
        torch.set_num_threads(threads)
        runtimes = {
            "pytorch_fp32": lambda q, t: score_cross_encoder_pairs(q, t, model_name=args.model, device="cpu", batch_size=args.batch_size),
            "onnx_fp32": lambda q, t: score_onnx_pairs(q, t, str(args.onnx_dir / FP32_FILENAME), args.batch_size, threads),
            "onnx_int8": lambda q, t: score_onnx_pairs(q, t, str(args.onnx_dir / INT8_FILENAME), args.batch_size, threads),
        }
        results = {name: run_runtime(workload, score, args.top_k) for name, score in runtimes.items()}
        for name in ("onnx_fp32", "onnx_int8"):
            results[name].update(compare_to(results["pytorch_fp32"], results[name]))
            results[name]["speedup_vs_pytorch_p50"] = round(results["pytorch_fp32"]["p50_ms"] / results[name]["p50_ms"], 2)
        report["runs"][f"{threads}_threads"] = {
            name: {key: value for key, value in result.items() if not key.startswith("_")} for name, result in results.items()
        }
    torch.set_num_threads(torch_threads_before)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
