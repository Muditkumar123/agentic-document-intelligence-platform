"""Tracked multi-hop evaluation: does the agent's retrieval loop find evidence across documents?

Runs the agent over ``data/eval/multihop_probes.jsonl`` (two-document questions)
and over the answerable golden questions (single-hop regression check) with each
loop variant, on an index of the short eval corpus plus full-length documents:

- ``single_pass`` — no decomposition, no retries (the pre-loop pipeline)
- ``decompose`` — sub-question searches, no retries
- ``retry`` — coverage-driven follow-up searches, no decomposition
- ``full`` — both (the default agent)

Deterministic with the extractive writer and the rule-based decomposer, so it runs in
CI. ``--decomposer llm`` plus a reasoning model adds an LLM-decomposition variant, and
``--writer-model-name`` swaps the extractive writer for an LLM so answer-level coverage
reflects all the evidence rather than the extractive writer's top-3 sentence picks.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

from adip.agents.multihop_eval import (
    DEFAULT_CORPUS_DIRS,
    DEFAULT_PROBES_PATH,
    aggregate_golden,
    aggregate_multihop,
    build_corpus_index,
    read_probes,
    score_golden_case,
    score_multihop_case,
)
from adip.agents.runner import run_agent
from adip.mlops.tracking import start_run
from adip.rag.evaluate import read_golden

VARIANTS: dict[str, dict[str, Any]] = {
    "single_pass": {"decomposer": "none", "max_retrieval_retries": 0},
    "decompose": {"decomposer": "rules", "max_retrieval_retries": 0},
    "retry": {"decomposer": "none", "max_retrieval_retries": 2},
    "full": {"decomposer": "rules", "max_retrieval_retries": 2},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adip.mlops.run_multihop_eval",
        description="Evaluate the agent's retrieval loop on two-document questions and single-hop questions.",
    )
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES_PATH)
    parser.add_argument("--golden", type=Path, default=Path("data/eval/golden_qa.jsonl"))
    parser.add_argument("--corpus-dir", type=Path, action="append", dest="corpus_dirs", default=None)
    parser.add_argument("--work-dir", type=Path, default=Path("data/processed/multihop_eval"))
    parser.add_argument("--backend", choices=["tfidf", "dense", "hybrid"], default="tfidf")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--engine", choices=["auto", "langgraph", "sequential"], default="auto")
    parser.add_argument(
        "--decomposer",
        choices=["rules", "llm"],
        default="rules",
        help="`llm` adds an llm_full variant whose sub-questions come from the reasoning model.",
    )
    parser.add_argument("--reasoning-provider", choices=["huggingface", "openai_compatible"], default=None)
    parser.add_argument("--reasoning-model-name", default=None)
    parser.add_argument("--reasoning-model-profile", default=None)
    parser.add_argument("--reasoning-endpoint-url", default=None)
    parser.add_argument("--reasoning-api-key", default=None, help="Session-only; never logged.")
    parser.add_argument("--reasoning-max-new-tokens", type=int, default=1024)
    parser.add_argument("--reasoning-device", default=None)
    parser.add_argument("--writer-provider", choices=["huggingface", "openai_compatible"], default=None)
    parser.add_argument("--writer-model-name", default=None, help="Default: the deterministic extractive writer.")
    parser.add_argument("--writer-endpoint-url", default=None)
    parser.add_argument("--writer-api-key", default=None, help="Session-only; never logged.")
    parser.add_argument("--writer-max-new-tokens", type=int, default=512)
    parser.add_argument("--skip-golden", action="store_true", help="Only run the multi-hop probes (for slow LLM runs).")
    parser.add_argument("--metrics-output", type=Path, default=Path("data/monitoring/multihop_eval_metrics.json"))
    parser.add_argument("--report-output", type=Path, default=Path("data/monitoring/multihop_eval_report.json"))
    parser.add_argument("--run-dir", type=Path, default=Path("data/monitoring/mlops_runs"))
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default=None)
    return parser


def run_variant(
    name: str,
    options: dict[str, Any],
    index,
    probes: list[dict[str, Any]],
    golden: list[dict[str, Any]],
    top_k: int,
    engine: str,
    writer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    writer_options = writer or {"model_profile": "extractive_baseline"}

    def run(question: str):
        return run_agent(
            question,
            index,
            task="qa",
            top_k=top_k,
            engine=engine,
            **writer_options,
            **options,
        ).state

    multihop_cases = [score_multihop_case(probe, run(probe["question"])) for probe in probes]
    golden_cases = [score_golden_case(row, run(row["question"])) for row in golden]
    return {
        "variant": name,
        "options": {key: value for key, value in options.items() if "api_key" not in key},
        "writer": writer_options.get("model_name") or writer_options.get("model_profile"),
        "multihop": aggregate_multihop(multihop_cases),
        "golden": aggregate_golden(golden_cases) if golden_cases else {},
        "multihop_cases": [case.to_dict() for case in multihop_cases],
        "golden_cases": golden_cases,
        "elapsed_ms": (time.perf_counter() - started) * 1000,
    }


def flatten_metrics(results: list[dict[str, Any]], chunk_count: int) -> dict[str, float]:
    metrics: dict[str, float] = {"multihop_index_chunk_count": float(chunk_count)}
    for result in results:
        name = result["variant"]
        for key, value in result["multihop"].items():
            if key != "by_type":
                metrics[f"multihop_{name}_{key}"] = float(value)
        for key, value in result["golden"].items():
            metrics[f"golden_{name}_{key}"] = float(value)
    # How much each loop variant adds over the single-pass pipeline (what the CI gate checks).
    baseline = metrics.get("multihop_single_pass_full_support_rate")
    if baseline is not None:
        for result in results:
            name = result["variant"]
            if name != "single_pass":
                metrics[f"multihop_{name}_full_support_gain"] = metrics[f"multihop_{name}_full_support_rate"] - baseline
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    corpus_dirs = args.corpus_dirs or list(DEFAULT_CORPUS_DIRS)
    variants = {name: dict(VARIANTS[name]) for name in args.variants}
    if args.decomposer == "llm":
        if not (args.reasoning_model_name or args.reasoning_model_profile):
            raise SystemExit("--decomposer llm needs --reasoning-model-name or --reasoning-model-profile")
        variants["llm_full"] = {
            "decomposer": "llm",
            "max_retrieval_retries": 2,
            "reasoning_provider": args.reasoning_provider,
            "reasoning_model_name": args.reasoning_model_name,
            "reasoning_model_profile": args.reasoning_model_profile,
            "reasoning_endpoint_url": args.reasoning_endpoint_url,
            "reasoning_api_key": args.reasoning_api_key,
            "reasoning_max_new_tokens": args.reasoning_max_new_tokens,
            "reasoning_device": args.reasoning_device,
        }

    with start_run(
        "multihop_eval",
        run_dir=args.run_dir,
        enable_mlflow=args.enable_mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        tags={"pipeline": "agent", "backend": args.backend},
    ) as run:
        run.log_params(
            {
                "probes": str(args.probes),
                "golden": str(args.golden),
                "corpus_dirs": [str(path) for path in corpus_dirs],
                "backend": args.backend,
                "top_k": args.top_k,
                "variants": list(variants),
                "engine": args.engine,
                "decomposer_model": args.reasoning_model_name or args.reasoning_model_profile or "",
                "writer": args.writer_model_name or "extractive_baseline",
            }
        )
        writer = None
        if args.writer_model_name:
            writer = {
                "llm_provider": args.writer_provider or "openai_compatible",
                "model_profile": None,
                "model_name": args.writer_model_name,
                "endpoint_url": args.writer_endpoint_url,
                "api_key": args.writer_api_key,
                "max_new_tokens": args.writer_max_new_tokens,
            }
        index = build_corpus_index(corpus_dirs, args.work_dir, backend=args.backend)
        probes = read_probes(args.probes)
        golden = [] if args.skip_golden else [row for row in read_golden(args.golden) if row.get("answerable", True)]
        results = [
            run_variant(name, options, index, probes, golden, args.top_k, args.engine, writer)
            for name, options in variants.items()
        ]
        metrics = flatten_metrics(results, len(index.chunks))
        run.log_metrics(metrics)

        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        report = {
            "config": {
                "backend": args.backend,
                "top_k": args.top_k,
                "corpus_dirs": [str(path) for path in corpus_dirs],
                "chunk_count": len(index.chunks),
                "probe_count": len(probes),
                "golden_question_count": len(golden),
            },
            "variants": results,
        }
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
        run.log_artifact("metrics", args.metrics_output)
        run.log_artifact("report", args.report_output)

    print(format_table(results))
    print(json.dumps({"metrics_output": str(args.metrics_output), "mlops_run_path": str(run.run_path)}, indent=2))
    return 0


def format_table(results: list[dict[str, Any]]) -> str:
    header = (
        f"{'variant':<12} {'hop recall':>10} {'both hops':>10} {'answer both':>12} {'searches':>9} "
        f"{'| golden hit':>12} {'golden cov':>11} {'golden retry':>13} {'golden searches':>16}"
    )
    lines = [header, "-" * len(header)]
    empty_golden = {"hit_rate": float("nan"), "expected_coverage": float("nan"), "retry_rate": float("nan"),
                    "avg_retrieval_calls": float("nan")}
    for result in results:
        multihop, golden = result["multihop"], result["golden"] or empty_golden
        lines.append(
            f"{result['variant']:<12} {multihop['hop_recall']:>10.3f} {multihop['full_support_rate']:>10.3f} "
            f"{multihop['answer_full_coverage']:>12.3f} {multihop['avg_retrieval_calls']:>9.2f} "
            f"{golden['hit_rate']:>12.3f} {golden['expected_coverage']:>11.3f} {golden['retry_rate']:>13.3f} "
            f"{golden['avg_retrieval_calls']:>16.2f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
