"""LLM adapter interfaces and local fallback implementations."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol
from urllib import error, request

BRIEF_REQUEST_KEYWORDS = {
    "brief",
    "report",
    "summarize",
    "summary",
    "overview",
    "write up",
}

STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "from",
    "have",
    "into",
    "paper",
    "that",
    "their",
    "there",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}

NOISY_SENTENCE_PREFIXES = (
    "keywords:",
    "this work was supported",
    "http://",
    "issn",
    "received ",
    "copyright",
)

CONTRIBUTION_PHRASES = (
    "in this paper",
    "we firstly",
    "we train",
    "we perform",
    "we present",
    "we apply",
    "we extend",
    "results show",
    "to prove",
    "successfully recover",
)

# Instruction used when a hosted answer is truncated (finish_reason == "length")
# and we ask the model to continue. It is deliberately strict about *not* echoing
# meta-commentary ("Continuation:", "The last word was ...") or restarting a
# section, because some hosted models (e.g. Gemini) otherwise narrate the splice.
CONTINUATION_INSTRUCTION = (
    "Continue the answer from exactly where it stopped. "
    "Output only the missing text. Do not repeat any words you already wrote, "
    "do not restart or re-title any section or heading, and do not wrap the text "
    "in quotes. Do not add any preamble, acknowledgement, or meta-commentary such "
    "as 'Continuation:', 'The last word was ...', or 'continuing from where I "
    "stopped'. Keep every factual claim grounded in the provided evidence with "
    "citations."
)

# A continuation chunk often opens with conversational/meta filler that a model
# emits in response to a "continue" turn. These match only the *leading* lines of
# a continuation chunk and are stripped before splicing.
_CONTINUATION_META_PATTERNS = (
    re.compile(
        r"^(?:sure|certainly|of course|okay|ok|alright|got it|understood|"
        r"no problem|as requested)\b",
        re.IGNORECASE,
    ),
    re.compile(r"^(?:wait|hold on|let me|let'?s|i['’]?ll|i will|i'?m going to)\b", re.IGNORECASE),
    re.compile(r"^here(?:'s| is)\b.*\b(?:rest|continuation|remaining|continue)\b", re.IGNORECASE),
    re.compile(r"^(?:continuation|continued|continuing|resuming|picking up)\b", re.IGNORECASE),
    re.compile(
        r"^the\s+(?:last|final|previous|prior)\s+"
        r"(?:word|line|sentence|phrase|token|character|part)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:where (?:you|i) (?:stopped|left off)|pick(?:ing)? up where|seamless(?:ly)?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"^\(continued\)\s*$", re.IGNORECASE),
)

# Cap on how many leading meta lines we are willing to drop from a continuation
# chunk, so an overly broad pattern can never swallow real answer content.
_MAX_CONTINUATION_META_LINES = 6
_QUOTE_CHARS = "\"'“”‘’"


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    question: str
    task_type: str
    domain_preset: str
    evidence: list[dict[str, Any]]
    max_new_tokens: int = 512


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    model_provider: str
    model_name: str
    latency_ms: float
    input_token_count: int
    output_token_count: int
    raw: dict[str, Any] | None = None
    gpu_memory: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMAdapter(Protocol):
    model_provider: str
    model_name: str

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        ...


class GroundedExtractiveAdapter:
    """Deterministic local generator used as the default LLMOps-safe baseline."""

    model_provider = "local"
    model_name = "grounded-extractive-v1"

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        start = time.perf_counter()
        if request.task_type == "brief" or (
            request.task_type == "qa" and looks_like_brief_request(request.question)
        ):
            text = self._brief(request)
        elif request.task_type == "plan":
            text = self._plan(request)
        elif request.task_type == "verify":
            text = self._verify(request)
        else:
            text = self._qa(request)
        latency_ms = (time.perf_counter() - start) * 1000
        return GenerationResponse(
            text=text,
            model_provider=self.model_provider,
            model_name=self.model_name,
            latency_ms=latency_ms,
            input_token_count=count_tokens(request.prompt),
            output_token_count=count_tokens(text),
            raw={"adapter": "deterministic_extractive"},
        )

    def _qa(self, request: GenerationRequest) -> str:
        if not request.evidence:
            return (
                "The indexed documents do not contain enough evidence to answer this question."
            )

        query_terms = meaningful_terms(request.question)
        answer_lines = []
        for index, item in enumerate(request.evidence[:3], start=1):
            snippet = evidence_focus_sentence(item, query_terms, max_chars=260)
            answer_lines.append(f"- Evidence {index}: {snippet} ({item['citation']})")
        return (
            f"Question: {request.question}\n\n"
            "Answer:\n"
            f"{chr(10).join(answer_lines)}\n\n"
            "How to read this:\n"
            "- These bullets are extractive evidence, not free-form synthesis.\n"
            "- Use the Agent brief mode or a local LLM profile for richer prose.\n\n"
            "LLMOps note: this answer was generated by the deterministic grounded baseline."
        )

    def _brief(self, request: GenerationRequest) -> str:
        if not request.evidence:
            return (
                f"Research Brief: {request.question}\n\n"
                "Status: insufficient evidence.\n\n"
                "The indexed documents do not contain enough evidence for a grounded brief."
            )

        query_terms = meaningful_terms(request.question)
        summary_lines = []
        evidence_lines = []
        coverage: dict[str, list[str]] = {}

        for index, item in enumerate(request.evidence[:5], start=1):
            focused = evidence_focus_sentence(item, query_terms, max_chars=260)
            if index <= 3:
                summary_lines.append(f"- {focused} ({item['citation']})")
            evidence_lines.append(f"- Evidence {index}: {focused} ({item['citation']})")
            coverage.setdefault(item["filename"], []).append(item["citation"])

        coverage_lines = [
            f"- {filename}: {len(citations)} retrieved chunk(s), including {citations[0]}"
            for filename, citations in sorted(coverage.items())
        ]
        return (
            f"Research Brief: {request.question}\n\n"
            f"Domain preset: {request.domain_preset}\n\n"
            "Executive Summary:\n"
            f"{chr(10).join(summary_lines)}\n\n"
            "Key Evidence:\n"
            f"{chr(10).join(evidence_lines)}\n\n"
            "Source Coverage:\n"
            f"{chr(10).join(coverage_lines)}\n\n"
            "Verification Notes:\n"
            "- Every evidence line includes a retrieved citation.\n"
            f"- The brief used {len(request.evidence[:5])} retrieved chunk(s) from the current index.\n"
            "- This is a deterministic LLMOps baseline, so it favors traceable evidence over polished prose.\n\n"
            "Limitations:\n"
            "- It does not infer beyond the retrieved text.\n"
            "- Use a local Qwen or DeepSeek profile for more natural abstractive synthesis."
        )

    def _plan(self, request: GenerationRequest) -> str:
        return (
            "1. Classify the user request and choose the answer format.\n"
            "2. Retrieve the most relevant document chunks for the request.\n"
            "3. Verify whether the retrieved evidence can support the answer.\n"
            "4. Write a concise grounded response with citations."
        )

    def _verify(self, request: GenerationRequest) -> str:
        if not request.evidence:
            return (
                "Supported claims:\n"
                "- None. The indexed documents do not contain enough evidence.\n\n"
                "Missing evidence:\n"
                "- No retrieved chunks were available for verification.\n\n"
                "Verifier decision:\n"
                "- Insufficient evidence."
            )

        top = request.evidence[0]
        return (
            "Supported claims:\n"
            f"- The strongest retrieved evidence is from {top['filename']} and should be cited "
            f"as {top['citation']} ({top['citation']}).\n\n"
            "Missing evidence:\n"
            "- The verifier only checked the retrieved chunks, so uncovered documents may still contain more context.\n\n"
            "Verifier decision:\n"
            f"- Evidence is usable if the final answer stays within the retrieved text ({top['citation']})."
        )


class TransformersTextGenerationAdapter:
    """Optional Hugging Face text-generation adapter for local model experiments."""

    model_provider = "huggingface"

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        local_files_only: bool = True,
    ) -> None:
        if importlib.util.find_spec("transformers") is None:
            raise ImportError("transformers is required for the Hugging Face adapter")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        import torch

        from adip.serving.gpu import reset_torch_peak_memory, torch_gpu_memory_snapshot

        start = time.perf_counter()
        reset_torch_peak_memory(self.device)
        prompt_text = render_chat_prompt(self.tokenizer, request.prompt)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        new_tokens = outputs[0][prompt_tokens:]
        # Local reasoning models (e.g. DeepSeek-R1 distills) emit <think>...</think>
        # spans; strip them so only the final answer shows, matching the hosted path.
        text = strip_reasoning_blocks(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        latency_ms = (time.perf_counter() - start) * 1000
        return GenerationResponse(
            text=text,
            model_provider=self.model_provider,
            model_name=self.model_name,
            latency_ms=latency_ms,
            input_token_count=prompt_tokens,
            output_token_count=int(new_tokens.shape[-1]),
            raw={"adapter": "transformers_causal_lm"},
            gpu_memory=torch_gpu_memory_snapshot(self.device),
        )


class OpenAICompatibleChatAdapter:
    """Adapter for vLLM, SGLang, Ollama proxies, or hosted OpenAI-compatible APIs."""

    model_provider = "openai_compatible"

    def __init__(
        self,
        model_name: str,
        endpoint_url: str | None = None,
        api_key: str | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout_seconds: int = 120,
        max_continuations: int = 2,
    ) -> None:
        self.model_name = model_name
        base_url = endpoint_url or os.getenv("ADIP_OPENAI_BASE_URL")
        if not base_url:
            raise ValueError(
                "OpenAI-compatible serving requires `endpoint_url` or ADIP_OPENAI_BASE_URL."
            )
        self.endpoint_url = normalize_chat_endpoint(base_url)
        self.api_key = api_key or os.getenv("ADIP_OPENAI_API_KEY", "")
        self.extra_body = extra_body or {}
        self.timeout_seconds = timeout_seconds
        self.max_continuations = max_continuations

    def generate(self, request_payload: GenerationRequest) -> GenerationResponse:
        start = time.perf_counter()
        messages = [
            {
                "role": "system",
                "content": "You are a grounded document intelligence assistant.",
            },
            {"role": "user", "content": request_payload.prompt},
        ]
        response_payload = self._post_chat(messages, request_payload.max_new_tokens)
        text = extract_chat_completion_text(response_payload)
        finish_reasons = [finish_reason(response_payload)]
        usage = dict(response_payload.get("usage", {}))
        continuation_count = 0

        while finish_reasons[-1] == "length" and continuation_count < self.max_continuations:
            messages = [
                *messages,
                {"role": "assistant", "content": text},
                {"role": "user", "content": CONTINUATION_INSTRUCTION},
            ]
            try:
                continuation_payload = self._post_chat(messages, request_payload.max_new_tokens)
                continuation_text = clean_continuation_text(
                    extract_chat_completion_text(continuation_payload)
                )
            except RuntimeError:
                # The primary answer already exists; a failed splice should never
                # abort the whole request. Stop continuing and return what we have.
                break
            if not continuation_text:
                break
            text = append_continuation(text, continuation_text)
            finish_reasons.append(finish_reason(continuation_payload))
            usage = merge_usage(usage, continuation_payload.get("usage", {}))
            continuation_count += 1

        if continuation_count:
            # Stitched answers can still carry a duplicated restarted line; collapse
            # near-duplicate lines so the splice reads as one continuous answer.
            text = collapse_near_duplicate_lines(text)

        latency_ms = (time.perf_counter() - start) * 1000
        return GenerationResponse(
            text=text,
            model_provider=self.model_provider,
            model_name=self.model_name,
            latency_ms=latency_ms,
            input_token_count=int(usage.get("prompt_tokens") or count_tokens(request_payload.prompt)),
            output_token_count=int(usage.get("completion_tokens") or count_tokens(text)),
            raw={
                "adapter": "openai_compatible_chat",
                "endpoint_url": self.endpoint_url,
                "extra_body_keys": sorted(self.extra_body),
                "finish_reason": finish_reasons[-1],
                "finish_reasons": finish_reasons,
                "continuation_count": continuation_count,
                "usage": usage,
            },
        )

    def _post_chat(self, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        payload.update(self.extra_body)
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "agentic-document-intelligence-platform/0.1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = request.Request(
            self.endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Provider request failed: HTTP Error {exc.code}: {exc.reason}. "
                f"Endpoint: {self.endpoint_url}. Model: {self.model_name}. "
                f"Response: {compact_text(body, 1200)}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Provider request failed: {exc.reason}. Endpoint: {self.endpoint_url}."
            ) from exc
        return response_payload


def get_adapter(
    provider: str = "extractive",
    model_name: str | None = None,
    local_files_only: bool = True,
    endpoint_url: str | None = None,
    api_key: str | None = None,
    device: str = "cuda:0",
    extra_body: dict[str, Any] | None = None,
) -> LLMAdapter:
    if provider == "extractive":
        return GroundedExtractiveAdapter()
    if provider == "huggingface":
        if not model_name:
            raise ValueError("model_name is required for the Hugging Face adapter")
        return TransformersTextGenerationAdapter(
            model_name=model_name,
            device=device,
            local_files_only=local_files_only,
        )
    if provider == "openai_compatible":
        if not model_name:
            raise ValueError("model_name is required for the OpenAI-compatible adapter")
        return OpenAICompatibleChatAdapter(
            model_name=model_name,
            endpoint_url=endpoint_url,
            api_key=api_key,
            extra_body=extra_body,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


def extract_chat_completion_text(response_payload: dict[str, Any]) -> str:
    """Extract assistant text from common OpenAI-compatible response shapes."""
    choices = response_payload.get("choices") or []
    if choices:
        first_choice = choices[0] or {}
        message = first_choice.get("message") or first_choice.get("delta") or {}
        text = strip_reasoning_blocks(normalize_content_value(message.get("content")))
        if text:
            return text
        refusal = normalize_content_value(message.get("refusal"))
        if refusal:
            return refusal
        if normalize_content_value(message.get("reasoning_content")):
            raise RuntimeError(
                "Provider response only included reasoning content, not a final assistant answer. "
                "Increase Max Tokens, disable reasoning/thinking for the model if possible, or use "
                "a non-reasoning writer model."
            )
        finish_reason = first_choice.get("finish_reason")
        if finish_reason == "length":
            raise RuntimeError(
                "Provider returned no visible assistant text because it stopped with "
                "`finish_reason=length`. Increase Max Tokens or use a model with lower "
                "reasoning/thinking token overhead. "
                f"Response: {compact_text(json.dumps(response_payload, sort_keys=True), 1200)}"
            )

    candidates = response_payload.get("candidates") or []
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        text = strip_reasoning_blocks(normalize_content_value(parts))
        if text:
            return text

    raise RuntimeError(
        "Provider response did not include assistant text. "
        f"Response: {compact_text(json.dumps(response_payload, sort_keys=True), 1200)}"
    )


def finish_reason(response_payload: dict[str, Any]) -> str | None:
    choices = response_payload.get("choices") or []
    if not choices:
        return None
    return choices[0].get("finish_reason")


def merge_usage(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, (int, float)) and isinstance(merged.get(key), (int, float)):
            merged[key] += value
        else:
            merged[key] = value
    return merged


def extract_reasoning_tokens(usage: dict[str, Any]) -> int:
    """Estimate hidden-reasoning token usage from an OpenAI-compatible usage block.

    Thinking models (Gemini 2.5, DeepSeek-R1, o-series) may report the tokens they
    spent on internal reasoning under ``completion_tokens_details.reasoning_tokens``.
    Those tokens count against ``max_tokens`` but never appear in the visible answer.

    Gemini's OpenAI-compatible endpoint omits that nested breakdown; there the
    reasoning shows up only as the gap between ``total_tokens`` and
    ``prompt_tokens + completion_tokens``, so we fall back to that.
    """
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict):
        value = details.get("reasoning_tokens")
        if isinstance(value, (int, float)) and value > 0:
            return int(value)

    total = usage.get("total_tokens")
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if all(isinstance(value, (int, float)) for value in (total, prompt, completion)):
        gap = int(total) - int(prompt) - int(completion)
        if gap > 0:
            return gap
    return 0


def build_answer_warning(
    generation: GenerationResponse,
    max_new_tokens: int | None = None,
) -> str | None:
    """Explain a truncated answer so a stub never looks like a silent failure.

    Returns a user-facing note when the model stopped on ``finish_reason == "length"``
    (it ran out of token budget), calling out hidden-reasoning consumption when the
    provider reports it. Returns ``None`` when the answer finished normally.
    """
    raw = generation.raw or {}
    if raw.get("finish_reason") != "length":
        return None

    usage = raw.get("usage") or {}
    reasoning_tokens = extract_reasoning_tokens(usage)
    budget_phrase = f" of the {max_new_tokens}-token budget" if max_new_tokens else ""

    if reasoning_tokens:
        cause = (
            f"The model spent about {reasoning_tokens} token(s) on internal reasoning"
            f"{budget_phrase}, leaving too few for the visible answer."
        )
    else:
        cause = f"The answer reached the token limit{budget_phrase} before it finished."

    return (
        f"This answer was cut off. {cause} "
        "Raise Max Tokens (thinking models such as Gemini 2.5 and DeepSeek-R1 often "
        "need 8000+), or set the model's reasoning/thinking effort lower or off."
    )


def append_continuation(text: str, continuation: str) -> str:
    if not text:
        return continuation
    if not continuation:
        return text
    continuation = _drop_seam_overlap(text, continuation)
    if not continuation:
        return text
    if text[-1].isspace() or continuation[0] in ".,;:!?)]}":
        return f"{text}{continuation}"
    return f"{text} {continuation}"


def clean_continuation_text(continuation: str) -> str:
    """Strip the conversational/meta preamble a model emits when told to continue.

    Hosted models sometimes answer a "continue" turn with filler such as
    ``Wait, let's make sure the continuation is seamless. / The last word was
    "number". / Continuation:`` before the real text. Splicing that verbatim is
    what produced the garbled SIMON answer, so we drop the leading meta lines (and
    any wrapping quotes) while leaving genuine continuation content untouched.
    """
    text = strip_reasoning_blocks(continuation)
    if not text:
        return ""

    lines = text.splitlines()
    start = 0
    dropped = 0
    while start < len(lines):
        stripped = lines[start].strip()
        if not stripped:
            start += 1
            continue
        remainder = _strip_inline_continuation_label(stripped)
        if remainder is not None:
            lines = [remainder, *lines[start + 1 :]]
            start = 0
            break
        if dropped < _MAX_CONTINUATION_META_LINES and _looks_like_continuation_meta(stripped):
            start += 1
            dropped += 1
            continue
        break

    cleaned = "\n".join(lines[start:]).strip()
    return _strip_wrapping_quotes(cleaned)


def _looks_like_continuation_meta(line: str) -> bool:
    return any(pattern.search(line) for pattern in _CONTINUATION_META_PATTERNS)


def _strip_inline_continuation_label(line: str) -> str | None:
    """Return content after an inline ``Continuation:`` style label, else ``None``.

    Only fires when a label is immediately followed by real text on the same line
    (``Continuation: of rounds ...``). A bare ``Continuation:`` returns ``None`` so
    it falls through to the meta-line drop instead.
    """
    match = re.match(r"^(?:continuation|continued|continuing|resuming)\s*:\s+(?=\S)", line, re.IGNORECASE)
    if not match:
        return None
    return line[match.end() :].strip()


def _strip_wrapping_quotes(text: str) -> str:
    if text and text[0] in _QUOTE_CHARS:
        stripped = text[1:].lstrip()
        if stripped and stripped[-1] in _QUOTE_CHARS:
            stripped = stripped[:-1].rstrip()
        return stripped
    return text


def _drop_seam_overlap(text: str, continuation: str, max_tokens: int = 40) -> str:
    """Drop a continuation prefix that verbatim repeats the tail of ``text``.

    Requires a substantial (>= 3 token, >= 12 char) exact overlap so coincidental
    matches on short common words ("the", "of") are never trimmed.
    """
    text_tokens = text.split()
    cont_tokens = continuation.split()
    if not text_tokens or not cont_tokens:
        return continuation
    limit = min(max_tokens, len(text_tokens), len(cont_tokens))
    for size in range(limit, 2, -1):
        tail = [token.casefold() for token in text_tokens[-size:]]
        head = [token.casefold() for token in cont_tokens[:size]]
        if tail == head and sum(len(token) for token in head) >= 12:
            return _drop_leading_tokens(continuation, size)
    return continuation


def _drop_leading_tokens(text: str, count: int) -> str:
    remainder = text.lstrip()
    for _ in range(count):
        match = re.match(r"\s*\S+", remainder)
        if not match:
            return ""
        remainder = remainder[match.end() :]
    return remainder.lstrip()


def collapse_near_duplicate_lines(
    text: str,
    window: int = 6,
    min_chars: int = 40,
    threshold: float = 0.82,
) -> str:
    """Remove a long line that largely repeats a recent preceding line.

    Targets the restart artifact where a continuation re-emits an almost-identical
    paragraph/heading (e.g. ``### 3. of rounds, or input differences ...`` right
    after ``... of rounds, and input differences ...``). Short lines are left
    alone so headings like ``### 3. Source Coverage`` survive.
    """
    kept_lines: list[str] = []
    recent_tokens: list[set[str]] = []
    for line in text.splitlines():
        normalized = _normalize_for_compare(line)
        if len(normalized) < min_chars:
            kept_lines.append(line)
            continue
        tokens = set(normalized.split())
        if tokens and any(
            _token_containment(tokens, previous) >= threshold
            for previous in recent_tokens[-window:]
        ):
            continue
        kept_lines.append(line)
        recent_tokens.append(tokens)
    return "\n".join(kept_lines)


def _normalize_for_compare(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _token_containment(new_tokens: set[str], previous_tokens: set[str]) -> float:
    if not new_tokens:
        return 0.0
    return len(new_tokens & previous_tokens) / len(new_tokens)


def normalize_content_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "".join(parts).strip()
    return str(value).strip()


def strip_reasoning_blocks(text: str) -> str:
    # Remove well-formed <think>...</think> blocks.
    without_blocks = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    # Some reasoning models (e.g. DeepSeek-R1 distills) emit the reasoning WITHOUT an
    # opening tag, because the chat template already injected `<think>`. The completion
    # then looks like "reasoning ... </think> answer", leaving a stray closing tag.
    # Drop everything up to and including the first such close.
    if re.search(r"</think\s*>", without_blocks, flags=re.IGNORECASE):
        without_blocks = re.sub(
            r"^.*?</think\s*>", "", without_blocks, count=1, flags=re.IGNORECASE | re.DOTALL
        )
    return without_blocks.strip()


def count_tokens(text: str) -> int:
    return len(text.split())


def compact_text(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3].rstrip()}..."


def looks_like_brief_request(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in BRIEF_REQUEST_KEYWORDS)


def meaningful_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9_]+", text.lower())
        if len(token) > 3 and token not in STOPWORDS
    }


def evidence_focus_sentence(
    item: dict[str, Any],
    query_terms: set[str],
    max_chars: int,
) -> str:
    sentences = split_evidence_sentences(str(item["text"]))
    if not sentences:
        return compact_text(str(item["text"]), max_chars=max_chars)

    def score(sentence: str) -> tuple[int, int]:
        lowered = sentence.lower()
        terms = meaningful_terms(sentence)
        overlap = len(terms & query_terms)
        contribution_bonus = 3 if any(phrase in lowered for phrase in CONTRIBUTION_PHRASES) else 0
        noise_penalty = 8 if lowered.startswith(NOISY_SENTENCE_PREFIXES) else 0
        length_fit = -abs(min(len(sentence), max_chars) - 180)
        return (overlap + contribution_bonus - noise_penalty, length_fit)

    best = max(sentences, key=score)
    return compact_text(best, max_chars=max_chars)


def split_evidence_sentences(text: str) -> list[str]:
    normalized = clean_evidence_text(text)
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    sentences = [part.strip() for part in parts if len(part.strip()) >= 35]
    if sentences:
        return sentences
    if not normalized:
        return []
    return [normalized]


def clean_evidence_text(text: str) -> str:
    normalized = " ".join(text.split())
    if " Abstract " in normalized[:1400]:
        normalized = "Abstract: " + normalized.split(" Abstract ", 1)[1]

    cleanup_patterns = [
        r"^KSII TRANSACTIONS ON INTERNET AND INFORMATION SYSTEMS VOL\.\s*15,\s*NO\.\s*2,\s*(?:Feb\.|February)\s*2021\s*\d+\s*",
        r"^\d+\s+Tian et al\.: Deep Learning Assisted Differential Cryptanalysis for the Lightweight Cipher SIMON\s*",
        r"^Copyright.*?published February 28,\s*2021\s*",
    ]
    for pattern in cleanup_patterns:
        normalized = re.sub(pattern, "", normalized, flags=re.IGNORECASE)

    normalized = re.sub(
        r"^\d+(?:\.\d+)?\.?\s+[A-Z][A-Za-z -]{2,60}\s+(?=[A-Z])",
        "",
        normalized,
    )
    return normalized.strip()


def render_chat_prompt(tokenizer: Any, prompt: str) -> str:
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


def normalize_chat_endpoint(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/chat/completions"):
        return stripped
    if stripped.endswith("/v1"):
        return f"{stripped}/chat/completions"
    return f"{stripped}/v1/chat/completions"
