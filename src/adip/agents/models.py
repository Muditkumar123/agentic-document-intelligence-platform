"""State and trace models for the baseline agent workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from adip.rag.coverage import DEFAULT_COVERAGE_THRESHOLD

DEFAULT_MAX_RETRIEVAL_RETRIES = 2
DEFAULT_MAX_RETRIEVAL_CALLS = 8
DEFAULT_RETRIEVAL_TIME_BUDGET_MS = 10_000.0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TraceEvent:
    node_name: str
    status: str
    started_at: str
    ended_at: str
    duration_ms: float
    input_summary: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentState:
    run_id: str
    question: str
    requested_task: str = "auto"
    task_type: str = "qa"
    domain_preset: str = "general"
    top_k: int = 5
    document_filter: str | None = None
    llm_provider: str | None = None
    model_name: str | None = None
    model_profile: str | None = None
    model_profiles_path: str = "config/model_profiles.yaml"
    endpoint_url: str | None = None
    api_key: str | None = None
    device: str = "cuda:0"
    prompt_dir: str = "prompts"
    prompt_version: str | None = None
    max_new_tokens: int = 512
    reasoning_effort: str = "auto"
    reasoning_provider: str | None = None
    reasoning_model_name: str | None = None
    reasoning_model_profile: str | None = None
    reasoning_endpoint_url: str | None = None
    reasoning_api_key: str | None = None
    reasoning_device: str | None = None
    reasoning_prompt_version: str | None = None
    reasoning_max_new_tokens: int = 256
    use_reasoning_planner: bool = False
    # Retrieval loop: how the question is split, and how much re-searching is allowed.
    decomposer: str = "rules"
    max_retrieval_retries: int = DEFAULT_MAX_RETRIEVAL_RETRIES
    max_retrieval_calls: int = DEFAULT_MAX_RETRIEVAL_CALLS
    retrieval_time_budget_ms: float = DEFAULT_RETRIEVAL_TIME_BUDGET_MS
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD
    sub_questions: list[str] = field(default_factory=list)
    pending_queries: list[dict[str, Any]] = field(default_factory=list)
    retrieval_streams: list[dict[str, Any]] = field(default_factory=list)
    evidence_pool: dict[str, dict[str, Any]] = field(default_factory=dict)
    sub_question_support: list[dict[str, Any]] = field(default_factory=list)
    # One entry per follow-up round: why the verifier sent the run back to retrieval.
    retrieval_retries: list[dict[str, Any]] = field(default_factory=list)
    retrieval_elapsed_ms: float = 0.0
    plan: list[str] = field(default_factory=list)
    planning_notes: list[str] = field(default_factory=list)
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    verification_notes: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    final_answer: str = ""
    status: str = "created"
    metrics: dict[str, Any] = field(default_factory=dict)
    llmops: dict[str, Any] = field(default_factory=dict)
    planning_llmops: dict[str, Any] = field(default_factory=dict)
    reasoning_llmops: dict[str, Any] = field(default_factory=dict)
    trace: list[TraceEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload.get("api_key"):
            payload["api_key"] = "***"
        if payload.get("reasoning_api_key"):
            payload["reasoning_api_key"] = "***"
        payload["trace"] = [event.to_dict() for event in self.trace]
        # Every accepted search result; the evidence the writer used is already in `retrieved`.
        payload.pop("evidence_pool", None)
        return payload


@dataclass(frozen=True)
class AgentResult:
    state: AgentState
    trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_path": self.trace_path,
            "state": self.state.to_dict(),
        }
