"""Tracked answer-quality (generation) evaluation command.

For each golden question it retrieves evidence, generates a grounded answer with
the chosen writer (deterministic extractive baseline by default), scores the
answer for faithfulness / relevance / expected-fact coverage / citations, and
logs an MLOps run record with the aggregate metrics and a full report artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from adip.llmops.generation_eval import GenerationEvalReport, aggregate_eval, score_answer
from adip.llmops.nli import DEFAULT_NLI_MODEL, NLIEntailmentScorer
from adip.llmops.pipeline import generate_grounded_response
from adip.mlops.tracking import start_run
from adip.rag.evaluate import read_golden
from adip.rag.rerank import DEFAULT_CROSS_ENCODER_MODEL, rerank_results
from adip.rag.retriever import load_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adip.mlops.run_generation_eval",
        description="Evaluate generated answer quality over a golden Q&A set and log an MLOps run.",
    )
    parser.add_argument("--index", type=Path, default=Path("data/processed/vector_index"))
    parser.add_argument("--golden", type=Path, default=Path("data/reference/golden_qa.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--reranker", choices=["none", "lexical", "cross_encoder"], default="none")
    parser.add_argument("--rerank-weight", type=float, default=0.25)
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CROSS_ENCODER_MODEL)
    parser.add_argument("--cross-encoder-device", default=None)
    parser.add_argument("--cross-encoder-batch-size", type=int, default=16)
    parser.add_argument("--allow-reranker-download", action="store_true")
    # Writer selection (deterministic extractive baseline by default).
    parser.add_argument("--task", choices=["qa", "brief"], default="qa")
    parser.add_argument("--domain", default="general")
    parser.add_argument("--provider", choices=["extractive", "huggingface", "openai_compatible"], default=None)
    parser.add_argument("--model-profile", default="extractive_baseline")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--endpoint-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--reasoning-effort", choices=["auto", "none", "low", "medium", "high"], default="auto")
    # Abstention: refuse when confidence is below this value (disabled when unset).
    parser.add_argument("--abstention-threshold", type=float, default=None)
    parser.add_argument("--abstention-mode", choices=["score", "nli"], default="score")
    parser.add_argument("--nli-model", default=DEFAULT_NLI_MODEL)
    parser.add_argument("--nli-device", default=None)
    parser.add_argument("--allow-nli-download", action="store_true")
    # Scoring thresholds.
    parser.add_argument("--grounded-threshold", type=float, default=0.5)
    parser.add_argument("--substring-overlap-threshold", type=float, default=0.6)
    # Outputs / tracking.
    parser.add_argument("--report-output", type=Path, default=Path("data/monitoring/generation_eval_report.json"))
    parser.add_argument("--metrics-output", type=Path, default=Path("data/monitoring/generation_eval_metrics.json"))
    parser.add_argument("--run-dir", type=Path, default=Path("data/monitoring/mlops_runs"))
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default=None)
    return parser


def evaluate_answer_quality(args: argparse.Namespace) -> GenerationEvalReport:
    index = load_index(args.index)
    golden = read_golden(args.golden)
    resolved_candidate_k = args.candidate_k or args.top_k
    if resolved_candidate_k < args.top_k:
        raise ValueError("candidate_k must be greater than or equal to top_k")

    entailment_scorer = None
    if args.abstention_threshold is not None and args.abstention_mode == "nli":
        entailment_scorer = NLIEntailmentScorer(
            model_name=args.nli_model,
            device=args.nli_device,
            local_files_only=not args.allow_nli_download,
        )

    cases = []
    for row in golden:
        candidates = index.search(row["question"], top_k=resolved_candidate_k)
        retrieved = rerank_results(
            row["question"],
            candidates,
            reranker=args.reranker,
            top_k=args.top_k,
            original_score_weight=args.rerank_weight,
            cross_encoder_model=args.cross_encoder_model,
            cross_encoder_device=args.cross_encoder_device,
            cross_encoder_batch_size=args.cross_encoder_batch_size,
            cross_encoder_local_files_only=not args.allow_reranker_download,
        )
        result = generate_grounded_response(
            question=row["question"],
            task_type=args.task,
            domain_preset=args.domain,
            retrieved=[item.to_dict() for item in retrieved],
            provider=args.provider,
            model_name=args.model_name,
            model_profile_id=args.model_profile,
            endpoint_url=args.endpoint_url,
            api_key=args.api_key,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            reasoning_effort=args.reasoning_effort,
            abstention_threshold=args.abstention_threshold,
            entailment_scorer=entailment_scorer,
        )
        cases.append(
            score_answer(
                row["question"],
                result.answer,
                result.evidence,
                row.get("expected_substrings"),
                answerable=row.get("answerable", True),
                grounded_threshold=args.grounded_threshold,
                substring_overlap_threshold=args.substring_overlap_threshold,
            )
        )
    return aggregate_eval(cases)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with start_run(
        "generation_quality_eval",
        run_dir=args.run_dir,
        enable_mlflow=args.enable_mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        tags={"pipeline": "llmops", "eval": "generation_quality"},
    ) as run:
        run.log_params(
            {
                "index": str(args.index),
                "golden": str(args.golden),
                "top_k": args.top_k,
                "candidate_k": args.candidate_k or args.top_k,
                "reranker": args.reranker,
                "task": args.task,
                "domain": args.domain,
                "provider": args.provider or "",
                "model_profile": args.model_profile or "",
                "model_name": args.model_name or "",
                "reasoning_effort": args.reasoning_effort,
                "abstention_threshold": args.abstention_threshold if args.abstention_threshold is not None else "",
                "abstention_mode": args.abstention_mode,
                "nli_model": args.nli_model if args.abstention_mode == "nli" else "",
                "grounded_threshold": args.grounded_threshold,
                "substring_overlap_threshold": args.substring_overlap_threshold,
            }
        )
        report = evaluate_answer_quality(args)
        metrics = report.metrics()
        run.log_metrics(metrics)

        report_payload = {
            "config": {
                "index": str(args.index),
                "golden": str(args.golden),
                "top_k": args.top_k,
                "reranker": args.reranker,
                "task": args.task,
                "domain": args.domain,
                "model_profile": args.model_profile,
                "provider": args.provider,
                "grounded_threshold": args.grounded_threshold,
            },
            "report": report.to_dict(),
        }
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        run.log_artifact("generation_eval_report", args.report_output)
        run.log_artifact("generation_eval_metrics", args.metrics_output)

    payload = {
        "metrics": metrics,
        "report_output": str(args.report_output),
        "metrics_output": str(args.metrics_output),
        "mlops_run_path": str(run.run_path),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
