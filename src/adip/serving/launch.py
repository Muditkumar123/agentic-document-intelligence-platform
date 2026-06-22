"""Generate local serving launch plans for model profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from adip.config.model_profiles import DEFAULT_MODEL_PROFILE_PATH, load_model_profile


@dataclass(frozen=True)
class LaunchPlan:
    profile_id: str
    model_name: str
    provider: str
    role: str
    recommended_runtime: str
    commands: dict[str, str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_launch_plan(
    profile_id: str,
    profiles_path: Path = DEFAULT_MODEL_PROFILE_PATH,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> LaunchPlan:
    profile = load_model_profile(profile_id, profiles_path)
    max_model_len = min(profile.context_window, 32768)
    tensor_parallel_size = 2 if "32b" in profile_id.lower() or "v3" in profile_id.lower() else 1
    hosted_api = profile.provider == "openai_compatible" and not profile.local_files_only

    commands = {
        "project_transformers_generate": (
            "PYTHONPATH=src python -m adip.serving generate "
            f"--model-profile {profile_id} "
            "--prompt \"Summarize the indexed evidence.\""
        ),
        "project_openai_server": (
            "PYTHONPATH=src python -m adip.serving server "
            f"--model-profile {profile_id} --host {host} --port {port}"
        ),
    }

    if profile.provider == "huggingface" or (profile.provider == "openai_compatible" and profile.local_files_only):
        commands["vllm"] = (
            f"vllm serve {profile.model_name} "
            f"--host {host} --port {port} "
            f"--max-model-len {max_model_len} "
            "--dtype bfloat16 "
            f"--tensor-parallel-size {tensor_parallel_size}"
        )
        commands["sglang"] = (
            "python -m sglang.launch_server "
            f"--model-path {profile.model_name} "
            f"--host {host} --port {port} "
            f"--tp {tensor_parallel_size}"
        )
    if hosted_api:
        endpoint = profile.serving.get("endpoint_url", "")
        endpoint_env = profile.serving.get("endpoint_env", "ADIP_OPENAI_BASE_URL")
        api_key_env = profile.serving.get("api_key_env", "ADIP_OPENAI_API_KEY")
        commands["hosted_api_env"] = (
            f"export {endpoint_env}={endpoint}\n"
            f"export {api_key_env}=<your-api-key>"
        )

    recommended_runtime = "project_transformers_generate"
    if hosted_api:
        recommended_runtime = "hosted_api"
    elif profile.provider == "openai_compatible":
        recommended_runtime = "vllm_or_sglang"
    elif profile.provider == "huggingface":
        recommended_runtime = "transformers_or_vllm"

    notes = [
        f"Profile role: {profile.role}.",
        f"Quantization note: {profile.quantization}.",
        "The project commands default to local files only unless --allow-download is passed.",
    ]
    if tensor_parallel_size > 1:
        notes.append("This profile is configured for tensor parallel serving across multiple GPUs.")
    if profile.provider == "openai_compatible":
        if hosted_api:
            notes.append("Set the profile API key environment variable before running LLMOps or agents.")
        else:
            notes.append("Set ADIP_OPENAI_BASE_URL to point LLMOps/agents at the running server.")

    return LaunchPlan(
        profile_id=profile.profile_id,
        model_name=profile.model_name,
        provider=profile.provider,
        role=profile.role,
        recommended_runtime=recommended_runtime,
        commands=commands,
        notes=notes,
    )
