"""Tracked agent workflow command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from adip.agents.runner import run_agent_from_index_path
from adip.mlops.tracking import start_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adip.mlops.run_agent",
        description="Run the agent workflow and log an MLOps run record.",
    )
    parser.add_argument("--index", type=Path, default=Path("data/processed/vector_index"))
    parser.add_argument("--question", required=True)
    parser.add_argument("--task", choices=["auto", "qa", "brief"], default="auto")
    parser.add_argument("--domain", default="general")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--llm-provider", choices=["extractive", "huggingface", "openai_compatible"], default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--model-profile", default=None)
    parser.add_argument("--model-profiles", type=Path, default=Path("config/model_profiles.yaml"))
    parser.add_argument("--endpoint-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-dir", type=Path, default=Path("prompts"))
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--reasoning-provider", choices=["extractive", "huggingface", "openai_compatible"], default=None)
    parser.add_argument("--reasoning-model-name", default=None)
    parser.add_argument("--reasoning-model-profile", default=None)
    parser.add_argument("--reasoning-endpoint-url", default=None)
    parser.add_argument("--reasoning-api-key", default=None)
    parser.add_argument("--reasoning-device", default=None)
    parser.add_argument("--reasoning-prompt-version", default=None)
    parser.add_argument("--reasoning-max-new-tokens", type=int, default=256)
    parser.add_argument("--use-reasoning-planner", action="store_true")
    parser.add_argument("--trace-dir", type=Path, default=Path("data/monitoring/agent_traces"))
    parser.add_argument("--metrics-output", type=Path, default=Path("data/monitoring/agent_smoke_metrics.json"))
    parser.add_argument("--run-dir", type=Path, default=Path("data/monitoring/mlops_runs"))
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with start_run(
        "agent_workflow",
        run_dir=args.run_dir,
        enable_mlflow=args.enable_mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        tags={"pipeline": "agent", "domain": args.domain, "task": args.task},
    ) as run:
        run.log_params(
            {
                "index": str(args.index),
                "question": args.question,
                "task": args.task,
                "domain": args.domain,
                "top_k": args.top_k,
                "llm_provider": args.llm_provider,
                "model_name": args.model_name or "",
                "model_profile": args.model_profile or "",
                "model_profiles": str(args.model_profiles),
                "endpoint_url": args.endpoint_url or "",
                "device": args.device,
                "prompt_dir": str(args.prompt_dir),
                "prompt_version": args.prompt_version or "",
                "max_new_tokens": args.max_new_tokens,
                "reasoning_provider": args.reasoning_provider or "",
                "reasoning_model_name": args.reasoning_model_name or "",
                "reasoning_model_profile": args.reasoning_model_profile or "",
                "reasoning_endpoint_url": args.reasoning_endpoint_url or "",
                "reasoning_device": args.reasoning_device or args.device,
                "reasoning_prompt_version": args.reasoning_prompt_version or "",
                "reasoning_max_new_tokens": args.reasoning_max_new_tokens,
                "use_reasoning_planner": args.use_reasoning_planner,
            }
        )
        result = run_agent_from_index_path(
            question=args.question,
            index_path=args.index,
            task=args.task,
            domain_preset=args.domain,
            top_k=args.top_k,
            llm_provider=args.llm_provider,
            model_name=args.model_name,
            model_profile=args.model_profile,
            model_profiles_path=args.model_profiles,
            endpoint_url=args.endpoint_url,
            api_key=args.api_key,
            device=args.device,
            prompt_dir=args.prompt_dir,
            prompt_version=args.prompt_version,
            max_new_tokens=args.max_new_tokens,
            reasoning_provider=args.reasoning_provider,
            reasoning_model_name=args.reasoning_model_name,
            reasoning_model_profile=args.reasoning_model_profile,
            reasoning_endpoint_url=args.reasoning_endpoint_url,
            reasoning_api_key=args.reasoning_api_key,
            reasoning_device=args.reasoning_device,
            reasoning_prompt_version=args.reasoning_prompt_version,
            reasoning_max_new_tokens=args.reasoning_max_new_tokens,
            use_reasoning_planner=args.use_reasoning_planner,
            trace_dir=args.trace_dir,
        )
        run.log_metrics(
            metrics := {
                "trace_event_count": len(result.state.trace),
                "retrieved_count": result.state.metrics.get("retrieved_count", 0),
                "citation_count": result.state.metrics.get("citation_count", 0),
                "answer_char_count": result.state.metrics.get("answer_char_count", 0),
                "workflow_duration_ms": result.state.metrics.get("workflow_duration_ms", 0),
                "llm_input_token_count": result.state.metrics.get("llm_input_token_count", 0),
                "llm_output_token_count": result.state.metrics.get("llm_output_token_count", 0),
                "llm_latency_ms": result.state.metrics.get("llm_latency_ms", 0),
                "llm_citation_coverage": result.state.metrics.get("llm_citation_coverage", 0),
                "llm_gpu_allocated_mb": result.state.metrics.get("llm_gpu_allocated_mb", 0),
                "llm_gpu_reserved_mb": result.state.metrics.get("llm_gpu_reserved_mb", 0),
                "llm_gpu_max_allocated_mb": result.state.metrics.get("llm_gpu_max_allocated_mb", 0),
                "llm_gpu_max_reserved_mb": result.state.metrics.get("llm_gpu_max_reserved_mb", 0),
                "planning_llm_input_token_count": result.state.metrics.get("planning_llm_input_token_count", 0),
                "planning_llm_output_token_count": result.state.metrics.get("planning_llm_output_token_count", 0),
                "planning_llm_latency_ms": result.state.metrics.get("planning_llm_latency_ms", 0),
                "planning_llm_structured_output": result.state.metrics.get("planning_llm_structured_output", 0),
                "planning_llm_normalized_answer_char_count": result.state.metrics.get(
                    "planning_llm_normalized_answer_char_count",
                    0,
                ),
                "planning_llm_gpu_allocated_mb": result.state.metrics.get("planning_llm_gpu_allocated_mb", 0),
                "planning_llm_gpu_reserved_mb": result.state.metrics.get("planning_llm_gpu_reserved_mb", 0),
                "planning_llm_gpu_max_allocated_mb": result.state.metrics.get("planning_llm_gpu_max_allocated_mb", 0),
                "planning_llm_gpu_max_reserved_mb": result.state.metrics.get("planning_llm_gpu_max_reserved_mb", 0),
                "reasoning_llm_input_token_count": result.state.metrics.get("reasoning_llm_input_token_count", 0),
                "reasoning_llm_output_token_count": result.state.metrics.get("reasoning_llm_output_token_count", 0),
                "reasoning_llm_latency_ms": result.state.metrics.get("reasoning_llm_latency_ms", 0),
                "reasoning_llm_citation_coverage": result.state.metrics.get("reasoning_llm_citation_coverage", 0),
                "reasoning_llm_structured_output": result.state.metrics.get("reasoning_llm_structured_output", 0),
                "reasoning_llm_normalized_answer_char_count": result.state.metrics.get(
                    "reasoning_llm_normalized_answer_char_count",
                    0,
                ),
                "reasoning_llm_gpu_allocated_mb": result.state.metrics.get("reasoning_llm_gpu_allocated_mb", 0),
                "reasoning_llm_gpu_reserved_mb": result.state.metrics.get("reasoning_llm_gpu_reserved_mb", 0),
                "reasoning_llm_gpu_max_allocated_mb": result.state.metrics.get("reasoning_llm_gpu_max_allocated_mb", 0),
                "reasoning_llm_gpu_max_reserved_mb": result.state.metrics.get("reasoning_llm_gpu_max_reserved_mb", 0),
            }
        )
        run.log_params(
            {
                "resolved_prompt_version": result.state.llmops.get("prompt", {}).get("version", ""),
                "prompt_hash": result.state.llmops.get("prompt", {}).get("template_hash", ""),
                "resolved_model_name": result.state.llmops.get("generation", {}).get("model_name", ""),
                "resolved_model_profile": result.state.llmops.get("model_profile", {}).get("profile_id", ""),
                "resolved_planning_prompt_version": result.state.planning_llmops.get("prompt", {}).get("version", ""),
                "planning_prompt_hash": result.state.planning_llmops.get("prompt", {}).get("template_hash", ""),
                "resolved_planning_model_name": result.state.planning_llmops.get("generation", {}).get("model_name", ""),
                "resolved_reasoning_prompt_version": result.state.reasoning_llmops.get("prompt", {}).get("version", ""),
                "reasoning_prompt_hash": result.state.reasoning_llmops.get("prompt", {}).get("template_hash", ""),
                "resolved_reasoning_model_name": result.state.reasoning_llmops.get("generation", {}).get("model_name", ""),
                "resolved_reasoning_model_profile": result.state.reasoning_llmops.get("model_profile", {}).get("profile_id", ""),
            }
        )
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        run.log_artifact("metrics", args.metrics_output)
        if result.trace_path:
            run.log_artifact("agent_trace", result.trace_path)

    payload = {
        "agent_run_id": result.state.run_id,
        "agent_status": result.state.status,
        "mlops_run_path": str(run.run_path),
        "metrics_output": str(args.metrics_output),
        "trace_path": result.trace_path,
        "task_type": result.state.task_type,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(result.state.final_answer)
        print(f"\nMLOps run: {run.run_path}")
        if result.trace_path:
            print(f"AgentOps trace: {result.trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
