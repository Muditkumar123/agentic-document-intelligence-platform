"""Tracked LLMOps smoke-generation command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from adip.llmops.pipeline import generate_grounded_response, write_llmops_report
from adip.mlops.tracking import start_run
from adip.rag.retriever import load_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adip.mlops.run_llmops_smoke",
        description="Run prompt-versioned generation and log an MLOps/LLMOps record.",
    )
    parser.add_argument("--index", type=Path, default=Path("data/processed/vector_index"))
    parser.add_argument("--question", required=True)
    parser.add_argument("--task", choices=["qa", "brief"], default="qa")
    parser.add_argument("--domain", default="general")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model-profile", default=None)
    parser.add_argument("--model-profiles", type=Path, default=Path("config/model_profiles.yaml"))
    parser.add_argument("--provider", choices=["extractive", "huggingface", "openai_compatible"], default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--endpoint-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-dir", type=Path, default=Path("prompts"))
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--report-output", type=Path, default=Path("data/monitoring/llmops_smoke_report.json"))
    parser.add_argument("--metrics-output", type=Path, default=Path("data/monitoring/llmops_smoke_metrics.json"))
    parser.add_argument("--run-dir", type=Path, default=Path("data/monitoring/mlops_runs"))
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with start_run(
        "llmops_smoke",
        run_dir=args.run_dir,
        enable_mlflow=args.enable_mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        tags={"pipeline": "llmops", "provider": args.provider or "", "task": args.task},
    ) as run:
        run.log_params(
            {
                "index": str(args.index),
                "question": args.question,
                "task": args.task,
                "domain": args.domain,
                "top_k": args.top_k,
                "provider": args.provider,
                "model_name": args.model_name or "",
                "model_profile": args.model_profile or "",
                "model_profiles": str(args.model_profiles),
                "endpoint_url": args.endpoint_url or "",
                "device": args.device,
                "prompt_dir": str(args.prompt_dir),
                "prompt_version": args.prompt_version or "",
                "max_new_tokens": args.max_new_tokens,
            }
        )
        index = load_index(args.index)
        retrieved = [item.to_dict() for item in index.search(args.question, top_k=args.top_k)]
        result = generate_grounded_response(
            question=args.question,
            task_type=args.task,
            domain_preset=args.domain,
            retrieved=retrieved,
            provider=args.provider,
            model_name=args.model_name,
            model_profile_id=args.model_profile,
            model_profiles_path=args.model_profiles,
            endpoint_url=args.endpoint_url,
            api_key=args.api_key,
            device=args.device,
            prompt_dir=args.prompt_dir,
            prompt_version=args.prompt_version,
            max_new_tokens=args.max_new_tokens,
            local_files_only=False if args.allow_download else None,
        )
        metrics = result.metrics()
        run.log_metrics(metrics)
        run.log_params(
            {
                "resolved_prompt_version": result.prompt.version,
                "prompt_hash": result.prompt.template_hash,
                "model_provider": result.generation.model_provider,
                "resolved_model_name": result.generation.model_name,
                "resolved_model_profile": result.model_profile.get("profile_id", "") if result.model_profile else "",
            }
        )
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        write_llmops_report(result, args.report_output)
        run.log_artifact("metrics", args.metrics_output)
        run.log_artifact("llmops_report", args.report_output)

    payload = {
        "metrics": metrics,
        "metrics_output": str(args.metrics_output),
        "mlops_run_path": str(run.run_path),
        "report_output": str(args.report_output),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
