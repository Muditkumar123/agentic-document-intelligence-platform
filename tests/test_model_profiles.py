import os
from pathlib import Path

import pytest

from adip.config.env import load_env_file
from adip.config.model_profiles import (
    load_model_profile,
    load_model_profiles,
    profile_runtime_status,
    resolve_profile_api_key,
    resolve_profile_endpoint,
)
from adip.llmops.models import normalize_chat_endpoint


def test_load_model_profiles_contains_qwen_and_deepseek():
    profiles = load_model_profiles(Path("config/model_profiles.yaml"))

    assert "qwen3_8b_default" in profiles
    assert "deepseek_r1_distill_qwen_14b_reasoning" in profiles
    assert profiles["qwen3_8b_default"].provider == "huggingface"
    assert profiles["deepseek_r1_distill_qwen_14b_reasoning"].role == "reasoning"
    assert profiles["deepseek_v4_pro_cloud"].provider == "openai_compatible"


def test_load_unknown_model_profile_lists_available_profiles():
    with pytest.raises(KeyError, match="Available profiles"):
        load_model_profile("missing_profile", Path("config/model_profiles.yaml"))


def test_normalize_chat_endpoint():
    assert normalize_chat_endpoint("http://localhost:8000") == (
        "http://localhost:8000/v1/chat/completions"
    )
    assert normalize_chat_endpoint("http://localhost:8000/v1") == (
        "http://localhost:8000/v1/chat/completions"
    )
    assert normalize_chat_endpoint("http://localhost:8000/v1/chat/completions") == (
        "http://localhost:8000/v1/chat/completions"
    )


def test_env_file_loader_reads_keys_without_overriding_existing_values(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "DEEPSEEK_API_KEY=test-secret # local only",
                "QUOTED_VALUE='two words'",
                "export EXISTING_VALUE=replace-me",
                "BAD-KEY=ignored",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("QUOTED_VALUE", raising=False)
    monkeypatch.setenv("EXISTING_VALUE", "keep-me")

    loaded = load_env_file(env_path)

    assert loaded["DEEPSEEK_API_KEY"] == "test-secret"
    assert os.environ["DEEPSEEK_API_KEY"] == "test-secret"
    assert os.environ["QUOTED_VALUE"] == "two words"
    assert os.environ["EXISTING_VALUE"] == "keep-me"
    assert "BAD-KEY" not in loaded


def test_deepseek_profile_runtime_status_uses_key_env_without_leaking_value(monkeypatch):
    profile = load_model_profile("deepseek_v4_pro_cloud", Path("config/model_profiles.yaml"))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    missing = profile_runtime_status(profile)

    assert missing["api_key_env"] == "DEEPSEEK_API_KEY"
    assert missing["api_key_configured"] is False
    assert resolve_profile_endpoint(profile) == "https://api.deepseek.com/chat/completions"

    monkeypatch.setenv("DEEPSEEK_API_KEY", "do-not-print-this")
    ready = profile_runtime_status(profile)

    assert ready["api_key_configured"] is True
    assert resolve_profile_api_key(profile) == "do-not-print-this"
    assert "do-not-print-this" not in str(ready)
