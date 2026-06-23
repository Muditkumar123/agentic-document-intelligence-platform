"""LLMOps generation pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from adip.config.env import load_project_env
from adip.config.model_profiles import (
    DEFAULT_MODEL_PROFILE_PATH,
    ModelProfile,
    load_model_profile,
    resolve_profile_api_key,
    resolve_profile_endpoint,
)
from adip.llmops.evaluation import LLMQualityReport, evaluate_generation
from adip.llmops.models import (
    GenerationRequest,
    GenerationResponse,
    GroundedExtractiveAdapter,
    build_answer_warning,
    get_adapter,
)
from adip.llmops.nli import EntailmentScorer
from adip.llmops.prompts import PromptTemplate, load_prompt_template
from adip.llmops.verifier import normalize_verifier_output

DOMAIN_FOCUS = {
    "academic": ["problem", "method", "dataset", "result", "limitation"],
    "finance": ["market", "metric", "risk", "growth", "revenue"],
    "crypto": ["protocol", "token", "security", "governance", "consensus"],
    "legal": ["party", "obligation", "deadline", "clause", "compliance"],
    "technical": ["system", "architecture", "dependency", "deployment", "interface"],
    "general": ["claim", "evidence", "risk", "open question"],
}


@dataclass(frozen=True)
class LLMOpsResult:
    answer: str
    prompt: PromptTemplate
    generation: GenerationResponse
    quality: LLMQualityReport
    evidence: list[dict[str, Any]]
    model_profile: dict[str, Any] | None = None
    structured_output: dict[str, Any] | None = None
    answer_warning: str | None = None

    def metrics(self) -> dict[str, float]:
        metrics = {
            "llm_input_token_count": float(self.generation.input_token_count),
            "llm_output_token_count": float(self.generation.output_token_count),
            "llm_latency_ms": float(self.generation.latency_ms),
            "llm_citation_coverage": float(self.quality.citation_coverage),
            "llm_visible_citation_count": float(self.quality.visible_citation_count),
            "llm_unsupported_sentence_count": float(self.quality.unsupported_sentence_count),
            "llm_answer_sentence_count": float(self.quality.answer_sentence_count),
        }
        if self.generation.gpu_memory:
            gpu_memory = self.generation.gpu_memory
            for key in ("allocated_mb", "reserved_mb", "max_allocated_mb", "max_reserved_mb"):
                if key in gpu_memory:
                    metrics[f"llm_gpu_{key}"] = float(gpu_memory[key])
        if self.structured_output:
            metrics["llm_structured_output"] = 1.0 if self.structured_output["structured"] else 0.0
            metrics["llm_normalized_answer_char_count"] = float(
                len(self.structured_output["final_text"])
            )
        return metrics

    def metadata(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt.to_dict(),
            "generation": self.generation.to_dict(),
            "quality": self.quality.to_dict(),
            "model_profile": self.model_profile,
            "structured_output": self.structured_output,
            "answer_warning": self.answer_warning,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["prompt"] = self.prompt.to_dict()
        payload["generation"] = self.generation.to_dict()
        payload["quality"] = self.quality.to_dict()
        return payload


def generate_grounded_response(
    question: str,
    task_type: str,
    domain_preset: str,
    retrieved: list[dict[str, Any]],
    provider: str | None = None,
    model_name: str | None = None,
    model_profile_id: str | None = None,
    model_profiles_path: Path = DEFAULT_MODEL_PROFILE_PATH,
    endpoint_url: str | None = None,
    api_key: str | None = None,
    device: str = "cuda:0",
    prompt_dir: Path = Path("prompts"),
    prompt_version: str | None = None,
    max_new_tokens: int | None = None,
    local_files_only: bool | None = None,
    reasoning_effort: str | None = None,
    abstention_threshold: float | None = None,
    entailment_scorer: EntailmentScorer | None = None,
) -> LLMOpsResult:
    load_project_env()
    model_profile = resolve_model_profile(model_profile_id, model_profiles_path)
    resolved_provider = provider or (model_profile.provider if model_profile else "extractive")
    resolved_model_name = model_name or (model_profile.model_name if model_profile else None)
    resolved_max_new_tokens = (
        max_new_tokens
        if max_new_tokens is not None
        else (model_profile.max_new_tokens if model_profile else 512)
    )
    resolved_local_files_only = (
        local_files_only
        if local_files_only is not None
        else (model_profile.local_files_only if model_profile else True)
    )
    resolved_endpoint_url = resolve_profile_endpoint(model_profile, endpoint_url)
    resolved_api_key = resolve_profile_api_key(model_profile, api_key)
    extra_body = dict(model_profile.serving.get("extra_body", {})) if model_profile else {}
    if reasoning_effort and reasoning_effort != "auto":
        # "none" is a real value here (it disables thinking on Gemini's OpenAI-compatible
        # endpoint); only "auto" means "leave the provider default untouched".
        extra_body["reasoning_effort"] = reasoning_effort

    evidence = build_evidence(retrieved)
    prompt = load_prompt_template(task_type=task_type, prompt_dir=prompt_dir, version=prompt_version)

    abstention = evaluate_abstention(
        evidence, abstention_threshold, entailment_scorer=entailment_scorer, question=question
    )
    if abstention is not None:
        # Evidence is too weak to ground an answer: refuse before spending a model
        # call, regardless of provider. This is a system-level RAG guardrail.
        quality = evaluate_generation(abstention.text, evidence)
        return LLMOpsResult(
            answer=abstention.text,
            prompt=prompt,
            generation=abstention,
            quality=quality,
            evidence=evidence,
            model_profile=model_profile.to_dict() if model_profile else None,
        )

    rendered_prompt = prompt.render(
        question=question,
        domain_preset=domain_preset,
        focus_areas=", ".join(DOMAIN_FOCUS.get(domain_preset, DOMAIN_FOCUS["general"])),
        evidence=evidence,
    )
    if should_use_no_evidence_fallback(task_type, evidence, resolved_provider):
        adapter = GroundedExtractiveAdapter()
    else:
        adapter = get_adapter(
            provider=resolved_provider,
            model_name=resolved_model_name,
            local_files_only=resolved_local_files_only,
            endpoint_url=resolved_endpoint_url,
            api_key=resolved_api_key,
            device=device,
            extra_body=extra_body,
        )
    generation = adapter.generate(
        GenerationRequest(
            prompt=rendered_prompt,
            question=question,
            task_type=task_type,
            domain_preset=domain_preset,
            evidence=evidence,
            max_new_tokens=resolved_max_new_tokens,
        )
    )
    structured_output = None
    quality_text = generation.text
    if task_type == "verify":
        verifier_output = normalize_verifier_output(generation.text, evidence)
        structured_output = verifier_output.to_dict()
        quality_text = verifier_output.final_text

    quality = evaluate_generation(quality_text, evidence)
    return LLMOpsResult(
        answer=generation.text,
        prompt=prompt,
        generation=generation,
        quality=quality,
        evidence=evidence,
        model_profile=model_profile.to_dict() if model_profile else None,
        structured_output=structured_output,
        answer_warning=build_answer_warning(generation, resolved_max_new_tokens),
    )


def resolve_model_profile(
    model_profile_id: str | None,
    model_profiles_path: Path,
) -> ModelProfile | None:
    if not model_profile_id:
        return None
    return load_model_profile(model_profile_id, model_profiles_path)


ABSTENTION_TEXT = (
    "The retrieved evidence is insufficient evidence to answer this question "
    "confidently, so no grounded answer was produced."
)


def evaluate_abstention(
    evidence: list[dict[str, Any]],
    abstention_threshold: float | None,
    *,
    entailment_scorer: EntailmentScorer | None = None,
    question: str | None = None,
) -> GenerationResponse | None:
    """Decide whether to abstain because the evidence is too weak to ground an answer.

    Returns a refusal ``GenerationResponse`` when the confidence is below
    ``abstention_threshold`` (or there is no evidence), otherwise ``None`` to
    proceed with normal generation. Disabled when the threshold is ``None``.

    Confidence is the maximum retrieval score across evidence (``score`` mode), or
    the maximum answer-entailment probability from ``entailment_scorer`` when one
    is supplied (``nli`` mode). The refusal text contains "insufficient evidence"
    so the existing refusal detector and refusal metrics pick it up.
    """
    if abstention_threshold is None:
        return None
    if entailment_scorer is not None:
        mode = "nli"
        confidence = float(entailment_scorer(question or "", evidence))
    else:
        mode = "score"
        confidence = max((float(item.get("score", 0.0)) for item in evidence), default=0.0)
    if confidence >= abstention_threshold:
        return None
    return GenerationResponse(
        text=ABSTENTION_TEXT,
        model_provider="abstention",
        model_name=f"evidence-gate-{mode}",
        latency_ms=0.0,
        input_token_count=0,
        output_token_count=0,
        raw={
            "abstained": True,
            "abstention_mode": mode,
            "confidence": confidence,
            "abstention_threshold": float(abstention_threshold),
        },
    )


def build_evidence(retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence = []
    for item in retrieved:
        chunk = item["chunk"]
        evidence.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "filename": chunk["filename"],
                "page_number": chunk["page_number"],
                "citation": item["citation"],
                "score": item["score"],
                "text": chunk["text"],
            }
        )
    return evidence


def should_use_no_evidence_fallback(
    task_type: str,
    evidence: list[dict[str, Any]],
    provider: str,
) -> bool:
    return task_type in {"brief", "qa", "verify"} and not evidence and provider != "extractive"


def write_llmops_report(result: LLMOpsResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
