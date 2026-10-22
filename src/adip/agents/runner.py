"""Runner for the baseline agent workflow."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Callable

from adip.agents.graph import AgentRoutes, langgraph_available, run_agent_graph
from adip.agents.models import (
    DEFAULT_MAX_RETRIEVAL_CALLS,
    DEFAULT_MAX_RETRIEVAL_RETRIES,
    DEFAULT_RETRIEVAL_TIME_BUDGET_MS,
    AgentResult,
    AgentState,
    TraceEvent,
    utc_now_iso,
)
from adip.agents.nodes import (
    check_citations,
    plan_task,
    retrieve_evidence,
    route_intent,
    verify_evidence,
    write_response,
)
from adip.rag.coverage import DEFAULT_COVERAGE_THRESHOLD
from adip.rag.retriever import RagIndex, load_index
from adip.tracing import mark_error, set_attributes, span

AgentNode = Callable[[AgentState, RagIndex], AgentState]

DEFAULT_NODES: tuple[tuple[str, AgentNode], ...] = (
    ("intent_router", route_intent),
    ("planner", plan_task),
    ("retriever", retrieve_evidence),
    ("evidence_verifier", verify_evidence),
    ("writer", write_response),
    ("citation_checker", check_citations),
)



def route_after_verification(state: AgentState) -> str:
    """The retrieval loop: search again while the verifier has queued follow-up searches."""
    return "retriever" if state.pending_queries else "writer"


DEFAULT_ROUTES: AgentRoutes = {
    "evidence_verifier": (route_after_verification, ("retriever", "writer")),
}

SUPPORTED_ENGINES = {"auto", "langgraph", "sequential"}


def resolve_engine(engine: str) -> str:
    """Pick the execution engine: LangGraph when installed, sequential otherwise."""
    if engine not in SUPPORTED_ENGINES:
        raise ValueError(f"Unsupported agent engine: {engine}")
    if engine == "auto":
        return "langgraph" if langgraph_available() else "sequential"
    return engine


def run_agent_from_index_path(
    question: str,
    index_path: Path,
    task: str = "auto",
    domain_preset: str = "general",
    top_k: int = 5,
    document_filter: str | None = None,
    llm_provider: str | None = None,
    model_name: str | None = None,
    model_profile: str | None = None,
    model_profiles_path: Path = Path("config/model_profiles.yaml"),
    endpoint_url: str | None = None,
    api_key: str | None = None,
    device: str = "cuda:0",
    prompt_dir: Path = Path("prompts"),
    prompt_version: str | None = None,
    max_new_tokens: int = 512,
    reasoning_effort: str = "auto",
    reasoning_provider: str | None = None,
    reasoning_model_name: str | None = None,
    reasoning_model_profile: str | None = None,
    reasoning_endpoint_url: str | None = None,
    reasoning_api_key: str | None = None,
    reasoning_device: str | None = None,
    reasoning_prompt_version: str | None = None,
    reasoning_max_new_tokens: int = 256,
    use_reasoning_planner: bool = False,
    trace_dir: Path | None = None,
    engine: str = "auto",
    decomposer: str = "rules",
    max_retrieval_retries: int = DEFAULT_MAX_RETRIEVAL_RETRIES,
    max_retrieval_calls: int = DEFAULT_MAX_RETRIEVAL_CALLS,
    retrieval_time_budget_ms: float = DEFAULT_RETRIEVAL_TIME_BUDGET_MS,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    reranker: str = "none",
    rerank_model: str | None = None,
    rerank_candidate_k: int | None = None,
) -> AgentResult:
    index = load_index(index_path)
    return run_agent(
        question=question,
        index=index,
        task=task,
        domain_preset=domain_preset,
        top_k=top_k,
        document_filter=document_filter,
        llm_provider=llm_provider,
        model_name=model_name,
        model_profile=model_profile,
        model_profiles_path=model_profiles_path,
        endpoint_url=endpoint_url,
        api_key=api_key,
        device=device,
        prompt_dir=prompt_dir,
        prompt_version=prompt_version,
        max_new_tokens=max_new_tokens,
        reasoning_effort=reasoning_effort,
        reasoning_provider=reasoning_provider,
        reasoning_model_name=reasoning_model_name,
        reasoning_model_profile=reasoning_model_profile,
        reasoning_endpoint_url=reasoning_endpoint_url,
        reasoning_api_key=reasoning_api_key,
        reasoning_device=reasoning_device,
        reasoning_prompt_version=reasoning_prompt_version,
        reasoning_max_new_tokens=reasoning_max_new_tokens,
        use_reasoning_planner=use_reasoning_planner,
        trace_dir=trace_dir,
        index_path=str(index_path),
        engine=engine,
        decomposer=decomposer,
        max_retrieval_retries=max_retrieval_retries,
        max_retrieval_calls=max_retrieval_calls,
        retrieval_time_budget_ms=retrieval_time_budget_ms,
        coverage_threshold=coverage_threshold,
        reranker=reranker,
        rerank_model=rerank_model,
        rerank_candidate_k=rerank_candidate_k,
    )


def run_agent(
    question: str,
    index: RagIndex,
    task: str = "auto",
    domain_preset: str = "general",
    top_k: int = 5,
    document_filter: str | None = None,
    llm_provider: str | None = None,
    model_name: str | None = None,
    model_profile: str | None = None,
    model_profiles_path: Path = Path("config/model_profiles.yaml"),
    endpoint_url: str | None = None,
    api_key: str | None = None,
    device: str = "cuda:0",
    prompt_dir: Path = Path("prompts"),
    prompt_version: str | None = None,
    max_new_tokens: int = 512,
    reasoning_effort: str = "auto",
    reasoning_provider: str | None = None,
    reasoning_model_name: str | None = None,
    reasoning_model_profile: str | None = None,
    reasoning_endpoint_url: str | None = None,
    reasoning_api_key: str | None = None,
    reasoning_device: str | None = None,
    reasoning_prompt_version: str | None = None,
    reasoning_max_new_tokens: int = 256,
    use_reasoning_planner: bool = False,
    trace_dir: Path | None = None,
    index_path: str | None = None,
    nodes: tuple[tuple[str, AgentNode], ...] = DEFAULT_NODES,
    engine: str = "auto",
    decomposer: str = "rules",
    max_retrieval_retries: int = DEFAULT_MAX_RETRIEVAL_RETRIES,
    max_retrieval_calls: int = DEFAULT_MAX_RETRIEVAL_CALLS,
    retrieval_time_budget_ms: float = DEFAULT_RETRIEVAL_TIME_BUDGET_MS,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    reranker: str = "none",
    rerank_model: str | None = None,
    rerank_candidate_k: int | None = None,
    routes: AgentRoutes | None = None,
) -> AgentResult:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if max_retrieval_retries < 0:
        raise ValueError("max_retrieval_retries must be 0 or greater")
    if max_retrieval_calls < 0:
        raise ValueError("max_retrieval_calls must be 0 or greater")
    if retrieval_time_budget_ms < 0:
        raise ValueError("retrieval_time_budget_ms must be 0 or greater")
    routes = DEFAULT_ROUTES if routes is None else routes
    resolved_engine = resolve_engine(engine)

    state = AgentState(
        run_id=f"agent_{uuid.uuid4().hex[:12]}",
        question=question,
        requested_task=task,
        domain_preset=domain_preset,
        top_k=top_k,
        document_filter=document_filter,
        llm_provider=llm_provider,
        model_name=model_name,
        model_profile=model_profile,
        model_profiles_path=str(model_profiles_path),
        endpoint_url=endpoint_url,
        api_key=api_key,
        device=device,
        prompt_dir=str(prompt_dir),
        prompt_version=prompt_version,
        max_new_tokens=max_new_tokens,
        reasoning_effort=reasoning_effort,
        reasoning_provider=reasoning_provider,
        reasoning_model_name=reasoning_model_name,
        reasoning_model_profile=reasoning_model_profile,
        reasoning_endpoint_url=reasoning_endpoint_url,
        reasoning_api_key=reasoning_api_key,
        reasoning_device=reasoning_device,
        reasoning_prompt_version=reasoning_prompt_version,
        reasoning_max_new_tokens=reasoning_max_new_tokens,
        use_reasoning_planner=use_reasoning_planner,
        decomposer=decomposer,
        max_retrieval_retries=max_retrieval_retries,
        max_retrieval_calls=max_retrieval_calls,
        retrieval_time_budget_ms=retrieval_time_budget_ms,
        coverage_threshold=coverage_threshold,
        reranker=reranker,
        rerank_model=rerank_model,
        rerank_candidate_k=rerank_candidate_k,
        metrics={
            "index_backend": index.backend,
            "index_chunk_count": len(index.chunks),
            "index_vocabulary_size": index.vocabulary_size,
            "index_path": index_path,
            "document_filter": document_filter,
            "llm_provider": llm_provider,
            "model_name": model_name,
            "model_profile": model_profile,
            "device": device,
            "reasoning_provider": reasoning_provider,
            "reasoning_model_name": reasoning_model_name,
            "reasoning_model_profile": reasoning_model_profile,
            "reasoning_device": reasoning_device or device,
            "use_reasoning_planner": use_reasoning_planner,
            "workflow_engine": resolved_engine,
            "max_retrieval_retries": max_retrieval_retries,
            "max_retrieval_calls": max_retrieval_calls,
            "retrieval_time_budget_ms": retrieval_time_budget_ms,
            "coverage_threshold": coverage_threshold,
            "reranker": reranker,
            "rerank_model": rerank_model if reranker == "cross_encoder" else None,
        },
    )

    workflow_started = time.perf_counter()
    run_attributes = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": "adip",
        "adip.run_id": state.run_id,
        "adip.engine": resolved_engine,
        "adip.task": task,
        "adip.top_k": top_k,
        "adip.index.backend": index.backend,
        "adip.decomposer": decomposer,
        "adip.reranker": reranker,
        "adip.model_profile": model_profile,
    }
    with span("invoke_agent adip", run_attributes) as run_span:
        if resolved_engine == "langgraph":
            state = run_agent_graph(state, index, nodes, _run_node, routes)
        else:
            state = run_sequential(state, index, nodes, routes)
        set_attributes(
            run_span,
            {
                "adip.status": state.status,
                "adip.task_type": state.task_type,
                "adip.retrieval.round_count": state.metrics.get("retrieval_round_count", 0),
                "adip.retrieval.call_count": state.metrics.get("retrieval_call_count", 0),
                "adip.retrieval.retry_count": len(state.retrieval_retries),
                "adip.citation_count": len(state.citations),
            },
        )
        if state.status == "failed":
            mark_error(run_span, "agent run failed")

    state.metrics["workflow_duration_ms"] = (time.perf_counter() - workflow_started) * 1000
    trace_path = write_trace(state, trace_dir) if trace_dir else None
    return AgentResult(state=state, trace_path=trace_path)


def next_node_name(
    node_names: list[str],
    routes: AgentRoutes,
    current: str,
    state: AgentState,
) -> str | None:
    """Same routing as the LangGraph engine: a router's choice if valid, else the next node in order."""
    position = node_names.index(current)
    fallback = node_names[position + 1] if position + 1 < len(node_names) else None
    router, targets = routes.get(current, (None, ()))
    if router is not None:
        choice = router(state)
        if choice in targets and choice in node_names:
            return choice
    return fallback


def run_sequential(
    state: AgentState,
    index: RagIndex,
    nodes: tuple[tuple[str, AgentNode], ...],
    routes: AgentRoutes,
) -> AgentState:
    """Dependency-free engine: follows the same routes, including the retrieval loop."""
    node_names = [name for name, _ in nodes]
    node_by_name = dict(nodes)
    max_steps = len(node_names) + 2 * (state.max_retrieval_retries + 1) + 4
    current: str | None = node_names[0] if node_names else None
    steps = 0
    while current is not None:
        state = _run_node(current, node_by_name[current], state, index)
        steps += 1
        if state.status == "failed":
            break
        if steps >= max_steps:
            state.status = "failed"
            state.verification_notes.append(f"Workflow stopped after {steps} steps without finishing.")
            break
        current = next_node_name(node_names, routes, current, state)
    return state


def _run_node(node_name: str, node: AgentNode, state: AgentState, index: RagIndex) -> AgentState:
    started_at = utc_now_iso()
    start_time = time.perf_counter()
    input_summary = summarize_state(state)
    error = None
    status = "completed"

    with span(f"agent.node {node_name}", {"adip.node": node_name}) as node_span:
        try:
            state = node(state, index)
        except Exception as exc:  # pragma: no cover - defensive trace path
            status = "failed"
            error = str(exc)
            state.status = "failed"
            state.verification_notes.append(f"Node {node_name} failed: {exc}")
            mark_error(node_span, error)
        set_attributes(
            node_span,
            {
                "adip.status": state.status,
                "adip.retrieved_count": len(state.retrieved),
                "adip.retrieval.round_count": state.metrics.get("retrieval_round_count", 0),
            },
        )

    ended_at = utc_now_iso()
    duration_ms = (time.perf_counter() - start_time) * 1000
    state.trace.append(
        TraceEvent(
            node_name=node_name,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            input_summary=input_summary,
            output_summary=summarize_state(state),
            error=error,
        )
    )
    return state


def summarize_state(state: AgentState) -> dict[str, object]:
    return {
        "status": state.status,
        "task_type": state.task_type,
        "plan_step_count": len(state.plan),
        "planning_note_count": len(state.planning_notes),
        "retrieved_count": len(state.retrieved),
        "sub_question_count": len(state.sub_questions),
        "retrieval_round_count": state.metrics.get("retrieval_round_count", 0),
        "retrieval_call_count": state.metrics.get("retrieval_call_count", 0),
        "retrieval_retry_count": len(state.retrieval_retries),
        "pending_query_count": len(state.pending_queries),
        "citation_count": len(state.citations),
        "verification_note_count": len(state.verification_notes),
        "answer_char_count": len(state.final_answer),
        "llm_provider": state.llm_provider,
        "model_profile": state.model_profile,
        "prompt_version": state.llmops.get("prompt", {}).get("version"),
        "planning_prompt_version": state.planning_llmops.get("prompt", {}).get("version"),
        "reasoning_model_profile": state.reasoning_model_profile,
        "reasoning_prompt_version": state.reasoning_llmops.get("prompt", {}).get("version"),
    }


def write_trace(state: AgentState, trace_dir: Path) -> str:
    path = trace_dir.expanduser()
    path.mkdir(parents=True, exist_ok=True)
    trace_path = path / f"{state.run_id}.json"
    trace_path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return str(trace_path)
