"""LLM-as-judge answer scoring.

The deterministic generation eval scores faithfulness with a lexical token-overlap
proxy: cheap, reproducible, CI-safe, but blind to paraphrase and to subtle
unsupported claims. This module adds a model-based judge that scores the same
answers semantically, behind the same report shape, so the two can be contrasted
(see the ``gen_eval_judge_*`` and agreement metrics in ``generation_eval``).

The judge is opt-in and offline: it calls an OpenAI-compatible endpoint (hosted or
local), so it never runs in the deterministic CI gate. The eval pipeline depends
only on the ``JudgeScorer`` callable protocol, so tests inject a fake judge.
API keys are passed per call and never written to disk.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from adip.llmops.models import (
    GenerationRequest,
    OpenAICompatibleChatAdapter,
    strip_reasoning_blocks,
)
from adip.llmops.prompts import DEFAULT_PROMPT_DIR, load_prompt_template

DEFAULT_JUDGE_MAX_NEW_TOKENS = 512


@dataclass(frozen=True)
class JudgeVerdict:
    faithfulness: float
    relevance: float
    unsupported_claims: list[str]
    raw_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class JudgeScorer(Protocol):
    """Scores one (question, answer, evidence) triple; None when judging failed."""

    def __call__(
        self, question: str, answer: str, evidence: list[dict[str, Any]]
    ) -> JudgeVerdict | None: ...


def _clamp(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))


def parse_judge_verdict(text: str) -> JudgeVerdict | None:
    """Extract the judge's JSON verdict from model output.

    Tolerates reasoning blocks, markdown fences, and surrounding prose; returns
    None when no well-formed verdict can be recovered so callers can count the
    failure instead of crashing the eval.
    """
    cleaned = strip_reasoning_blocks(text or "")
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
        faithfulness = _clamp(payload["faithfulness"])
        relevance = _clamp(payload["relevance"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    claims = payload.get("unsupported_claims") or []
    if not isinstance(claims, list):
        claims = []
    return JudgeVerdict(
        faithfulness=faithfulness,
        relevance=relevance,
        unsupported_claims=[str(claim) for claim in claims],
        raw_text=text,
    )


class LLMJudge:
    """Judges answers with an OpenAI-compatible model using the versioned judge prompt."""

    def __init__(
        self,
        model_name: str,
        endpoint_url: str | None = None,
        api_key: str | None = None,
        prompt_dir: Path = DEFAULT_PROMPT_DIR,
        max_new_tokens: int = DEFAULT_JUDGE_MAX_NEW_TOKENS,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.prompt = load_prompt_template("judge", prompt_dir=prompt_dir)
        self.max_new_tokens = max_new_tokens
        self.adapter = OpenAICompatibleChatAdapter(
            model_name=model_name,
            endpoint_url=endpoint_url,
            api_key=api_key,
            extra_body=extra_body,
        )

    def __call__(
        self, question: str, answer: str, evidence: list[dict[str, Any]]
    ) -> JudgeVerdict | None:
        rendered = self.prompt.render(question=question, answer=answer, evidence=evidence)
        response = self.adapter.generate(
            GenerationRequest(
                prompt=rendered,
                question=question,
                task_type="judge",
                domain_preset="general",
                evidence=evidence,
                max_new_tokens=self.max_new_tokens,
            )
        )
        return parse_judge_verdict(response.text)
