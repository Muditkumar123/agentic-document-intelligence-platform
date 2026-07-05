"""Tracked LoRA fine-tuning experiment (ROADMAP Phase 7).

Builds the chunk-category dataset from the eval corpus, runs the deterministic
baselines, optionally trains the frozen-head reference and the LoRA-adapted
classifier, and logs everything — params, per-approach metrics, and the full
comparison report — through the standard MLOps run tracking (MLflow-ready).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from adip.finetuning.baselines import majority_baseline, tfidf_logreg_baseline
from adip.finetuning.dataset import (
    build_labeled_chunks,
    parse_sources_categories,
    split_by_document,
)
from adip.mlops.tracking import start_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adip.mlops.run_lora_experiment",
        description="Compare baselines vs a LoRA-adapted classifier on chunk categories and log an MLOps run.",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/eval/raw"))
    parser.add_argument("--sources", type=Path, default=Path("data/eval/SOURCES.md"))
    parser.add_argument("--chunk-size", type=int, default=40, help="Words per training sample.")
    parser.add_argument("--holdout-per-category", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    # LoRA settings (used unless --skip-lora).
    parser.add_argument("--base-model", default="distilroberta-base")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument(
        "--skip-lora",
        action="store_true",
        help="Run only the deterministic baselines (no torch/transformers/peft needed).",
    )
    parser.add_argument("--adapter-output", type=Path, default=Path("models/lora_chunk_classifier"))
    parser.add_argument("--report-output", type=Path, default=Path("data/monitoring/lora_experiment_report.json"))
    parser.add_argument("--run-dir", type=Path, default=Path("data/monitoring/mlops_runs"))
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with start_run(
        "lora_finetune_experiment",
        run_dir=args.run_dir,
        enable_mlflow=args.enable_mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        tags={"pipeline": "finetuning", "experiment": "chunk_category_lora"},
    ) as run:
        categories = parse_sources_categories(args.sources)
        chunks = build_labeled_chunks(args.raw_dir, categories, chunk_size=args.chunk_size)
        train, evaluation = split_by_document(
            chunks, holdout_per_category=args.holdout_per_category, seed=args.seed
        )

        run.log_params(
            {
                "raw_dir": str(args.raw_dir),
                "chunk_size_words": args.chunk_size,
                "holdout_per_category": args.holdout_per_category,
                "seed": args.seed,
                "train_size": len(train),
                "eval_size": len(evaluation),
                "category_count": len({chunk.label for chunk in chunks}),
                "base_model": args.base_model if not args.skip_lora else "",
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "skip_lora": args.skip_lora,
            }
        )

        results = [majority_baseline(train, evaluation), tfidf_logreg_baseline(train, evaluation, seed=args.seed)]
        if not args.skip_lora:
            from adip.finetuning.lora import train_lora_classifier

            shared = dict(
                base_model=args.base_model,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                device=args.device,
                seed=args.seed,
                local_files_only=not args.allow_model_download,
            )
            results.append(train_lora_classifier(train, evaluation, head_only=True, **shared))
            results.append(
                train_lora_classifier(
                    train,
                    evaluation,
                    head_only=False,
                    lora_r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    adapter_output=args.adapter_output,
                    **shared,
                )
            )

        metrics: dict[str, float] = {
            "train_size": float(len(train)),
            "eval_size": float(len(evaluation)),
        }
        for result in results:
            prefix = result["approach"]
            metrics[f"{prefix}_accuracy"] = result["accuracy"]
            metrics[f"{prefix}_macro_f1"] = result["macro_f1"]
        run.log_metrics(metrics)

        report = {
            "dataset": {
                "train_size": len(train),
                "eval_size": len(evaluation),
                "eval_documents": sorted({chunk.filename for chunk in evaluation}),
                "categories": sorted({chunk.label for chunk in chunks}),
                "chunk_size_words": args.chunk_size,
                "split": "document-level (no document appears in both train and eval)",
            },
            "results": results,
        }
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        run.log_artifact("lora_experiment_report", args.report_output)

    payload = {
        "metrics": metrics,
        "report_output": str(args.report_output),
        "mlops_run_path": str(run.run_path),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
