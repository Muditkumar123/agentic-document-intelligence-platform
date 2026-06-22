"""Serving environment inspection."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from adip.config.env import load_project_env
from adip.config.model_profiles import DEFAULT_MODEL_PROFILE_PATH, load_model_profiles


@dataclass(frozen=True)
class ServingEnvironment:
    python_packages: dict[str, bool]
    cuda_available: bool
    gpu_count: int
    gpu_names: list[str]
    model_cache: dict[str, dict[str, Any]]
    openai_base_url_set: bool
    openai_api_key_set: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_serving_environment(
    profiles_path: Path = DEFAULT_MODEL_PROFILE_PATH,
    hf_home: Path | None = None,
) -> ServingEnvironment:
    load_project_env()
    packages = package_availability(
        ["torch", "transformers", "accelerate", "bitsandbytes", "vllm", "fastapi", "uvicorn"]
    )
    cuda_available = False
    gpu_count = 0
    gpu_names: list[str] = []

    if packages["torch"]:
        import torch

        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(index) for index in range(gpu_count)]

    profiles = load_model_profiles(profiles_path)
    cache_root = resolve_hf_cache_root(hf_home)
    model_cache = {
        profile_id: inspect_model_cache(profile.model_name, cache_root)
        for profile_id, profile in profiles.items()
    }

    return ServingEnvironment(
        python_packages=packages,
        cuda_available=cuda_available,
        gpu_count=gpu_count,
        gpu_names=gpu_names,
        model_cache=model_cache,
        openai_base_url_set=bool(os.getenv("ADIP_OPENAI_BASE_URL")),
        openai_api_key_set=bool(os.getenv("ADIP_OPENAI_API_KEY")),
    )


def package_availability(package_names: list[str]) -> dict[str, bool]:
    return {package: importlib.util.find_spec(package) is not None for package in package_names}


def resolve_hf_cache_root(hf_home: Path | None = None) -> Path:
    if hf_home is not None:
        return hf_home.expanduser()
    if os.getenv("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    return Path("~/.cache/huggingface/hub").expanduser()


def inspect_model_cache(model_name: str, cache_root: Path) -> dict[str, Any]:
    cache_dir = cache_root / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = cache_dir / "snapshots"
    snapshots = []
    if snapshots_dir.exists():
        snapshots = sorted(path.name for path in snapshots_dir.iterdir() if path.is_dir())
    return {
        "model_name": model_name,
        "cache_dir": str(cache_dir),
        "cached": cache_dir.exists() and bool(snapshots),
        "snapshot_count": len(snapshots),
        "snapshots": snapshots[-3:],
    }
