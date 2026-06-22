"""Serving backends for local generation."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Protocol

from adip.config.model_profiles import ModelProfile
from adip.serving.gpu import reset_torch_peak_memory, torch_gpu_memory_snapshot


@dataclass(frozen=True)
class ServingResponse:
    text: str
    model_name: str
    model_provider: str
    latency_ms: float
    input_token_count: int
    output_token_count: int
    gpu_memory: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ServingBackend(Protocol):
    model_name: str
    model_provider: str

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> ServingResponse:
        ...


class ExtractiveServingBackend:
    """Small deterministic backend for health checks and CI."""

    model_provider = "local"

    def __init__(self, profile: ModelProfile) -> None:
        self.profile = profile
        self.model_name = profile.model_name

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> ServingResponse:
        del max_new_tokens
        start = time.perf_counter()
        clipped = " ".join(prompt.split())[:500]
        text = (
            "Local serving baseline response.\n\n"
            f"Profile: {self.profile.profile_id}\n"
            f"Prompt: {clipped}"
        )
        return ServingResponse(
            text=text,
            model_name=self.model_name,
            model_provider=self.model_provider,
            latency_ms=(time.perf_counter() - start) * 1000,
            input_token_count=count_tokens(prompt),
            output_token_count=count_tokens(text),
            gpu_memory=None,
        )


class TransformersServingBackend:
    """Transformers backend for cached or explicitly downloaded Hugging Face models."""

    model_provider = "huggingface"

    def __init__(
        self,
        profile: ModelProfile,
        device: str = "cuda:0",
        allow_download: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.profile = profile
        self.model_name = profile.model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        local_files_only = profile.local_files_only and not allow_download

        self.tokenizer = AutoTokenizer.from_pretrained(
            profile.model_name,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            profile.model_name,
            dtype=dtype,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.load_gpu_memory = torch_gpu_memory_snapshot(self.device)

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> ServingResponse:
        import torch

        start = time.perf_counter()
        reset_torch_peak_memory(self.device)
        prompt_text = render_chat_prompt(self.tokenizer, prompt)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens or self.profile.max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        new_tokens = outputs[0][prompt_tokens:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return ServingResponse(
            text=text,
            model_name=self.model_name,
            model_provider=self.model_provider,
            latency_ms=(time.perf_counter() - start) * 1000,
            input_token_count=prompt_tokens,
            output_token_count=int(new_tokens.shape[-1]),
            gpu_memory=torch_gpu_memory_snapshot(self.device),
        )


def load_serving_backend(
    profile: ModelProfile,
    device: str = "cuda:0",
    allow_download: bool = False,
) -> ServingBackend:
    if profile.provider == "extractive":
        return ExtractiveServingBackend(profile)
    if profile.provider == "huggingface":
        return TransformersServingBackend(profile, device=device, allow_download=allow_download)
    raise ValueError(
        f"Profile `{profile.profile_id}` uses provider `{profile.provider}`. "
        "Use a vLLM/SGLang/OpenAI-compatible server for this profile."
    )


def count_tokens(text: str) -> int:
    return len(text.split())


def render_chat_prompt(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return prompt
