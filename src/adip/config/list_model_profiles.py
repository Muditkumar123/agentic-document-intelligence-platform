"""CLI for listing configured model profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from adip.config.model_profiles import DEFAULT_MODEL_PROFILE_PATH, load_model_profiles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adip.config.list_model_profiles",
        description="List configured LLM model profiles.",
    )
    parser.add_argument("--profiles", type=Path, default=DEFAULT_MODEL_PROFILE_PATH)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profiles = load_model_profiles(args.profiles)
    payload = {profile_id: profile.to_dict() for profile_id, profile in profiles.items()}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for profile_id, profile in profiles.items():
            print(f"{profile_id}: {profile.model_name} [{profile.provider}, {profile.role}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
