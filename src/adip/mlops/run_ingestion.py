"""Tracked ingestion command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from adip.ingestion.pipeline import ingest_path
from adip.mlops.fingerprint import directory_fingerprint
from adip.mlops.tracking import start_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adip.mlops.run_ingestion",
        description="Run ingestion and log an MLOps run record.",
    )
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument(
        "--parser",
        choices=["default", "unstructured"],
        default="default",
        help="Document parser. `unstructured` extracts tables (requires the [tables] extra).",
    )
    parser.add_argument("--metrics-output", type=Path, default=Path("data/monitoring/ingestion_metrics.json"))
    parser.add_argument("--run-dir", type=Path, default=Path("data/monitoring/mlops_runs"))
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with start_run(
        "document_ingestion",
        run_dir=args.run_dir,
        enable_mlflow=args.enable_mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        tags={"pipeline": "ingestion"},
    ) as run:
        input_fingerprint = directory_fingerprint(args.input)
        run.log_params(
            {
                "input": str(args.input),
                "output": str(args.output),
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap,
                "parser": args.parser,
                "input_file_count": len(input_fingerprint),
            }
        )
        result = ingest_path(
            input_path=args.input,
            output_path=args.output,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            parser=args.parser,
        )
        metrics = {
            "document_count": result.document_count,
            "page_count": result.page_count,
            "chunk_count": result.chunk_count,
        }
        run.log_metrics(metrics)
        run.log_artifact("chunks_jsonl", args.output)
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        run.log_artifact("metrics", args.metrics_output)

    payload = result.to_dict()
    payload["mlops_run_path"] = str(run.run_path)
    payload["metrics_output"] = str(args.metrics_output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
