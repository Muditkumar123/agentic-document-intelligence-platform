"""Model profile registry for local and API-backed LLM serving."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MODEL_PROFILE_PATH = Path("config/model_profiles.yaml")
DEFAULT_OPENAI_BASE_URL_ENV = "ADIP_OPENAI_BASE_URL"
DEFAULT_OPENAI_API_KEY_ENV = "ADIP_OPENAI_API_KEY"


@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    description: str
    role: str
    provider: str
    model_name: str
    context_window: int
    max_new_tokens: int
    quantization: str = "none"
    local_files_only: bool = True
    recommended_for: list[str] = field(default_factory=list)
    serving: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_model_profiles(path: Path = DEFAULT_MODEL_PROFILE_PATH) -> dict[str, ModelProfile]:
    profile_path = path.expanduser()
    if not profile_path.exists():
        raise FileNotFoundError(f"Model profile file not found: {profile_path}")

    payload = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    raw_profiles = payload.get("profiles", {})
    profiles: dict[str, ModelProfile] = {}
    for profile_id, raw in raw_profiles.items():
        profiles[profile_id] = ModelProfile(
            profile_id=profile_id,
            description=raw["description"],
            role=raw["role"],
            provider=raw["provider"],
            model_name=raw["model_name"],
            context_window=int(raw["context_window"]),
            max_new_tokens=int(raw["max_new_tokens"]),
            quantization=raw.get("quantization", "none"),
            local_files_only=bool(raw.get("local_files_only", True)),
            recommended_for=list(raw.get("recommended_for", [])),
            serving=dict(raw.get("serving", {})),
        )
    return profiles


def load_model_profile(
    profile_id: str,
    path: Path = DEFAULT_MODEL_PROFILE_PATH,
) -> ModelProfile:
    profiles = load_model_profiles(path)
    if profile_id not in profiles:
        available = ", ".join(sorted(profiles))
        raise KeyError(f"Unknown model profile `{profile_id}`. Available profiles: {available}")
    return profiles[profile_id]


def resolve_profile_endpoint(
    profile: ModelProfile | None,
    endpoint_url: str | None = None,
) -> str | None:
    if endpoint_url:
        return endpoint_url
    if profile is not None:
        endpoint_env = profile.serving.get("endpoint_env")
        if endpoint_env and os.getenv(str(endpoint_env)):
            return os.environ[str(endpoint_env)]
        if profile.serving.get("endpoint_url"):
            return str(profile.serving["endpoint_url"])
    return os.getenv(DEFAULT_OPENAI_BASE_URL_ENV)


def resolve_profile_api_key(
    profile: ModelProfile | None,
    api_key: str | None = None,
) -> str | None:
    if api_key:
        return api_key
    if profile is not None:
        api_key_env = profile.serving.get("api_key_env")
        if api_key_env and os.getenv(str(api_key_env)):
            return os.environ[str(api_key_env)]
    return os.getenv(DEFAULT_OPENAI_API_KEY_ENV)


def profile_runtime_status(profile: ModelProfile) -> dict[str, Any]:
    endpoint_env = str(profile.serving.get("endpoint_env") or DEFAULT_OPENAI_BASE_URL_ENV)
    api_key_env = str(profile.serving.get("api_key_env") or DEFAULT_OPENAI_API_KEY_ENV)
    endpoint_url = resolve_profile_endpoint(profile)
    api_key = resolve_profile_api_key(profile)
    return {
        "provider": profile.provider,
        "endpoint_env": endpoint_env,
        "endpoint_configured": bool(endpoint_url),
        "api_key_env": api_key_env,
        "api_key_configured": bool(api_key),
        "uses_api_key": profile.provider == "openai_compatible",
    }
