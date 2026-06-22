"""CLI for local model serving utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from adip.config.model_profiles import DEFAULT_MODEL_PROFILE_PATH, load_model_profile
from adip.serving.backends import load_serving_backend
from adip.serving.environment import inspect_serving_environment
from adip.serving.launch import build_launch_plan
from adip.serving.openai_server import run_openai_compatible_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adip.serving",
        description="Inspect and run local LLM serving utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect CUDA/packages/model cache.")
    inspect_parser.add_argument("--profiles", type=Path, default=DEFAULT_MODEL_PROFILE_PATH)
    inspect_parser.add_argument("--json", action="store_true")

    plan_parser = subparsers.add_parser("launch-plan", help="Print launch commands for a profile.")
    plan_parser.add_argument("--model-profile", required=True)
    plan_parser.add_argument("--profiles", type=Path, default=DEFAULT_MODEL_PROFILE_PATH)
    plan_parser.add_argument("--host", default="0.0.0.0")
    plan_parser.add_argument("--port", type=int, default=8000)
    plan_parser.add_argument("--json", action="store_true")

    generate_parser = subparsers.add_parser("generate", help="Generate with a local profile backend.")
    generate_parser.add_argument("--model-profile", default="extractive_baseline")
    generate_parser.add_argument("--profiles", type=Path, default=DEFAULT_MODEL_PROFILE_PATH)
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--device", default="cuda:0")
    generate_parser.add_argument("--max-new-tokens", type=int, default=None)
    generate_parser.add_argument("--allow-download", action="store_true")
    generate_parser.add_argument("--json", action="store_true")

    server_parser = subparsers.add_parser("server", help="Start a small OpenAI-compatible server.")
    server_parser.add_argument("--model-profile", default="extractive_baseline")
    server_parser.add_argument("--profiles", type=Path, default=DEFAULT_MODEL_PROFILE_PATH)
    server_parser.add_argument("--host", default="127.0.0.1")
    server_parser.add_argument("--port", type=int, default=8000)
    server_parser.add_argument("--device", default="cuda:0")
    server_parser.add_argument("--allow-download", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        return inspect_command(args)
    if args.command == "launch-plan":
        return launch_plan_command(args)
    if args.command == "generate":
        return generate_command(args)
    if args.command == "server":
        return server_command(args)
    raise ValueError(f"Unknown command: {args.command}")


def inspect_command(args: argparse.Namespace) -> int:
    environment = inspect_serving_environment(args.profiles)
    payload = environment.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"CUDA available: {payload['cuda_available']}")
        print(f"GPU count: {payload['gpu_count']}")
        for index, name in enumerate(payload["gpu_names"]):
            print(f"GPU {index}: {name}")
        print("Packages:")
        for package, available in payload["python_packages"].items():
            print(f"  {package}: {available}")
        print("Model cache:")
        for profile_id, cache in payload["model_cache"].items():
            print(f"  {profile_id}: cached={cache['cached']} snapshots={cache['snapshot_count']}")
    return 0


def launch_plan_command(args: argparse.Namespace) -> int:
    plan = build_launch_plan(args.model_profile, args.profiles, host=args.host, port=args.port)
    payload = plan.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Profile: {plan.profile_id}")
        print(f"Model: {plan.model_name}")
        print(f"Recommended runtime: {plan.recommended_runtime}")
        print("Commands:")
        for name, command in plan.commands.items():
            print(f"\n[{name}]\n{command}")
        print("\nNotes:")
        for note in plan.notes:
            print(f"- {note}")
    return 0


def generate_command(args: argparse.Namespace) -> int:
    profile = load_model_profile(args.model_profile, args.profiles)
    backend = load_serving_backend(
        profile,
        device=args.device,
        allow_download=args.allow_download,
    )
    response = backend.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    if args.json:
        print(json.dumps(response.to_dict(), indent=2, sort_keys=True))
    else:
        print(response.text)
    return 0


def server_command(args: argparse.Namespace) -> int:
    profile = load_model_profile(args.model_profile, args.profiles)
    backend = load_serving_backend(
        profile,
        device=args.device,
        allow_download=args.allow_download,
    )
    run_openai_compatible_server(backend, profile, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
