"""RAG query CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from adip.rag.answer import build_extractive_answer
from adip.rag.rerank import DEFAULT_CROSS_ENCODER_MODEL, rerank_results
from adip.rag.retriever import load_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adip.rag.query",
        description="Search a local RAG index and return cited evidence.",
    )
    parser.add_argument(
        "--index",
        "-x",
        type=Path,
        default=Path("data/processed/vector_index"),
        help="Directory containing the saved RAG index.",
    )
    parser.add_argument(
        "--question",
        "-q",
        required=True,
        help="Question to search for.",
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
    parser.add_argument("--rerank-weight", type=float, default=0.25)
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CROSS_ENCODER_MODEL)
    parser.add_argument("--cross-encoder-device", default=None)
    parser.add_argument("--cross-encoder-batch-size", type=int, default=16)
    parser.add_argument("--allow-reranker-download", action="store_true")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON instead of a readable answer.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    index = load_index(args.index)
    candidate_k = args.candidate_k or args.top_k
    if candidate_k < args.top_k:
        raise ValueError("candidate_k must be greater than or equal to top_k")
    candidates = index.search(args.question, top_k=candidate_k)
    retrieved = rerank_results(
        args.question,
        candidates,
        reranker=args.reranker,
        top_k=args.top_k,
        original_score_weight=args.rerank_weight,
        cross_encoder_model=args.cross_encoder_model,
        cross_encoder_device=args.cross_encoder_device,
        cross_encoder_batch_size=args.cross_encoder_batch_size,
        cross_encoder_local_files_only=not args.allow_reranker_download,
    )
    answer = build_extractive_answer(args.question, retrieved)
    payload = {
        "question": args.question,
        "answer": answer,
        "candidate_k": candidate_k,
        "cross_encoder_model": args.cross_encoder_model if args.reranker == "cross_encoder" else None,
        "reranker": args.reranker,
        "retrieved": [item.to_dict() for item in retrieved],
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
