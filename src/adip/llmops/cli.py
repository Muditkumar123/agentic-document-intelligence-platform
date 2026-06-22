"""LLMOps smoke-generation CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from adip.llmops.pipeline import generate_grounded_response, write_llmops_report
from adip.rag.retriever import load_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adip.llmops",
        description="Run a prompt-versioned grounded generation smoke test.",
    )
    parser.add_argument("--index", type=Path, default=Path("data/processed/vector_index"))
    parser.add_argument("--question", required=True)
    parser.add_argument("--task", choices=["qa", "brief"], default="qa")
    parser.add_argument("--domain", default="general")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--model-profile",
        default=None,
        help="Named model profile from config/model_profiles.yaml.",
    )
    parser.add_argument(
        "--model-profiles",
        type=Path,
        default=Path("config/model_profiles.yaml"),
        help="Model profile YAML path.",
    )
    parser.add_argument(
        "--provider",
        choices=["extractive", "huggingface", "openai_compatible"],
        default=None,
        help="Override generation provider.",
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--endpoint-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-dir", type=Path, default=Path("prompts"))
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/monitoring/llmops_smoke_report.json"))
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
    write_llmops_report(result, args.output)
    payload = {
        "answer": result.answer,
        "metrics": result.metrics(),
        "model_provider": result.generation.model_provider,
        "model_name": result.generation.model_name,
        "prompt_version": result.prompt.version,
        "prompt_hash": result.prompt.template_hash,
        "model_profile": result.model_profile,
        "quality": result.quality.to_dict(),
        "report_path": str(args.output),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(result.answer)
        print(f"\nLLMOps report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
