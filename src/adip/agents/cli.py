"""Command-line interface for the baseline agent workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from adip.agents.runner import run_agent_from_index_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adip.agents",
        description="Run the baseline agent workflow over a local RAG index.",
    )
    parser.add_argument(
        "--index",
        "-x",
        type=Path,
        default=Path("data/processed/vector_index"),
        help="Directory containing the saved RAG index.",
    )
    parser.add_argument("--question", "-q", required=True, help="User question or task.")
    parser.add_argument(
        "--task",
        choices=["auto", "qa", "brief"],
        default="auto",
        help="Agent task type. `auto` routes from the question text.",
    )
    parser.add_argument(
        "--domain",
        default="general",
        help="Domain preset such as general, academic, finance, crypto, legal, or technical.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--llm-provider",
        choices=["extractive", "huggingface", "openai_compatible"],
        default=None,
        help="Generation backend for the writer node.",
    )
    parser.add_argument("--model-name", default=None, help="Model name for provider-specific adapters.")
    parser.add_argument("--model-profile", default=None, help="Named model profile to use.")
    parser.add_argument(
        "--model-profiles",
        type=Path,
        default=Path("config/model_profiles.yaml"),
        help="Model profile YAML path.",
    )
    parser.add_argument("--endpoint-url", default=None, help="OpenAI-compatible endpoint override.")
    parser.add_argument("--api-key", default=None, help="OpenAI-compatible API key override.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-dir", type=Path, default=Path("prompts"))
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--reasoning-provider",
        choices=["extractive", "huggingface", "openai_compatible"],
        default=None,
        help="Optional generation backend for the evidence verifier node.",
    )
    parser.add_argument("--reasoning-model-name", default=None)
    parser.add_argument("--reasoning-model-profile", default=None)
    parser.add_argument("--reasoning-endpoint-url", default=None)
    parser.add_argument("--reasoning-api-key", default=None)
    parser.add_argument("--reasoning-device", default=None)
    parser.add_argument("--reasoning-prompt-version", default=None)
    parser.add_argument("--reasoning-max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--use-reasoning-planner",
        action="store_true",
        help="Use the reasoning model in the planner node before retrieval.",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("data/monitoring/agent_traces"),
        help="Directory where AgentOps traces are written.",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Do not persist an AgentOps trace JSON file.",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON output.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    trace_dir = None if args.no_trace else args.trace_dir
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
        trace_dir=trace_dir,
    )

    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(result.state.final_answer)
        if result.trace_path:
            print(f"\nAgentOps trace: {result.trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
