"""RAG index builder CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from adip.rag.chunks import read_chunks_jsonl
from adip.rag.retriever import build_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adip.rag.index",
        description="Build a local vector index from ingestion chunks.",
    )
    parser.add_argument(
        "--chunks",
        "-c",
        type=Path,
        default=Path("data/processed/chunks.jsonl"),
        help="Input ingestion chunks JSONL.",
    )
    parser.add_argument(
        "--index",
        "-x",
        type=Path,
        default=Path("data/processed/vector_index"),
        help="Output directory for the saved index.",
    )
    parser.add_argument(
        "--backend",
        choices=["tfidf", "dense", "dense_lsa", "sentence_transformers", "hybrid"],
        default="tfidf",
        help="Embedding/index backend. `hybrid` fuses BM25 and dense rankings with RRF.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        help="Maximum n-gram length for the TF-IDF baseline.",
    )
    parser.add_argument(
        "--embedding-model",
        default="lsa",
        help="Dense embedding model. Use `lsa` for dependency-light dense retrieval.",
    )
    parser.add_argument(
        "--dense-dimensions",
        type=int,
        default=128,
        help="Target dense dimensions for the LSA dense retriever.",
    )
    parser.add_argument(
        "--no-faiss",
        action="store_true",
        help="Disable FAISS even when it is installed.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="Reciprocal-rank-fusion constant for the hybrid backend.",
    )
    parser.add_argument(
        "--hybrid-dense-weight",
        type=float,
        default=0.5,
        help="Dense-ranking weight (0..1) for the hybrid backend; BM25 gets the rest.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    chunks = read_chunks_jsonl(args.chunks)
    index = build_index(
        chunks,
        backend=args.backend,
        ngram_max=args.ngram_max,
        embedding_model=args.embedding_model,
        dense_dimensions=args.dense_dimensions,
        use_faiss=not args.no_faiss,
        rrf_k=args.rrf_k,
        hybrid_dense_weight=args.hybrid_dense_weight,
    )
    index.save(args.index)
    print(
        json.dumps(
            {
                "backend": index.backend,
                "chunk_count": len(index.chunks),
                "embedding_model": index.embedding_model,
                "index_path": str(args.index),
                "metadata": index.metadata,
                "source_chunks": str(args.chunks),
                "vocabulary_size": index.vocabulary_size,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
