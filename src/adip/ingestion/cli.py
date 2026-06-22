"""Command-line interface for document ingestion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from adip.ingestion.pipeline import ingest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adip.ingestion",
        description="Parse documents, chunk text, and write traceable JSONL chunks.",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("data/raw"),
        help="Input file or directory. Supported: .pdf, .txt, .md.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/processed/chunks.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Maximum number of whitespace tokens per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=120,
        help="Number of whitespace tokens to overlap between adjacent chunks.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = ingest_path(
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0
