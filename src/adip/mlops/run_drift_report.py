"""Tracked drift baseline builder and drift report command.

``--rebuild-baseline`` runs every golden question through the current index and
summarizes the reference distribution (vocabulary, question lengths, top
retrieval scores). Without it, the command compares the logged live queries
against the existing baseline and writes a drift report. Both paths log through
the standard MLOps run tracking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from adip.mlops.tracking import start_run
from adip.monitoring.drift import (
    DEFAULT_BASELINE,
    DEFAULT_QUERY_LOG,
    QueryRecord,
    build_baseline,
    drift_report,
    load_baseline,
    read_query_log,
)
from adip.rag.evaluate import read_golden
from adip.rag.retriever import load_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adip.mlops.run_drift_report",
        description="Build the drift baseline from golden questions or report drift of logged queries.",
    )
    parser.add_argument("--index", type=Path, default=Path("data/processed/vector_index"))
    parser.add_argument("--golden", type=Path, default=Path("data/eval/golden_qa.jsonl"))
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--query-log", type=Path, default=DEFAULT_QUERY_LOG)
    parser.add_argument("--rebuild-baseline", action="store_true")
    parser.add_argument("--report-output", type=Path, default=Path("data/monitoring/drift_report.json"))
    parser.add_argument("--log-limit", type=int, default=500)
    parser.add_argument("--run-dir", type=Path, default=Path("data/monitoring/mlops_runs"))
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default=None)
    return parser


def baseline_records_from_golden(index_path: Path, golden_path: Path) -> list[QueryRecord]:
    index = load_index(index_path)
    records: list[QueryRecord] = []
    for row in read_golden(golden_path):
        if not row.get("answerable", True):
            continue  # unanswerable probes are off-distribution by design
        retrieved = index.search(row["question"], top_k=1)
        top_score = retrieved[0].score if retrieved else 0.0
        records.append(QueryRecord(question=row["question"], top_score=top_score))
    return records


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with start_run(
        "drift_report",
        run_dir=args.run_dir,
        enable_mlflow=args.enable_mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        tags={"pipeline": "monitoring", "report": "drift"},
    ) as run:
        run.log_params(
            {
                "index": str(args.index),
                "golden": str(args.golden),
                "baseline": str(args.baseline),
                "query_log": str(args.query_log),
                "rebuild_baseline": args.rebuild_baseline,
                "log_limit": args.log_limit,
            }
        )

        if args.rebuild_baseline:
            records = baseline_records_from_golden(args.index, args.golden)
            baseline = build_baseline(records)
            args.baseline.parent.mkdir(parents=True, exist_ok=True)
            args.baseline.write_text(json.dumps(baseline, indent=2, sort_keys=True), encoding="utf-8")
            run.log_metrics({"baseline_query_count": float(baseline["query_count"])})
            run.log_artifact("drift_baseline", args.baseline)
            payload = {
                "action": "baseline_rebuilt",
                "baseline": str(args.baseline),
                "baseline_query_count": baseline["query_count"],
                "mlops_run_path": str(run.run_path),
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0

        baseline = load_baseline(args.baseline)
        if baseline is None:
            raise FileNotFoundError(
                f"No drift baseline at {args.baseline}; run with --rebuild-baseline first"
            )
        recent = read_query_log(args.query_log, limit=args.log_limit)
        report = drift_report(baseline, recent)
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        run.log_artifact("drift_report", args.report_output)
        if report.get("available"):
            run.log_metrics(
                {
                    "recent_query_count": float(report["recent_query_count"]),
                    "vocabulary_oov_rate": report["components"]["vocabulary_oov_rate"]["value"],
                    "question_length_z": report["components"]["question_length_z"]["value"],
                    "retrieval_score_psi": report["components"]["retrieval_score_psi"]["value"],
                }
            )

    payload = {
        "action": "drift_report",
        "report_output": str(args.report_output),
        "overall_status": report.get("overall_status", "unavailable"),
        "mlops_run_path": str(run.run_path),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
