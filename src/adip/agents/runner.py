"""Runner for the baseline agent workflow."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Callable

from adip.agents.graph import langgraph_available, run_agent_graph
from adip.agents.models import AgentResult, AgentState, TraceEvent, utc_now_iso
from adip.agents.nodes import (
    check_citations,
    plan_task,
    retrieve_evidence,
    route_intent,
    verify_evidence,
    write_response,
)
from adip.rag.retriever import RagIndex, load_index

AgentNode = Callable[[AgentState, RagIndex], AgentState]

DEFAULT_NODES: tuple[tuple[str, AgentNode], ...] = (
    ("intent_router", route_intent),
    ("planner", plan_task),
    ("retriever", retrieve_evidence),
    ("evidence_verifier", verify_evidence),
    ("writer", write_response),
    ("citation_checker", check_citations),
)

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
) -> AgentResult:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
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
        },
    )

    workflow_started = time.perf_counter()
    if resolved_engine == "langgraph":
        state = run_agent_graph(state, index, nodes, _run_node)
    else:
        for node_name, node in nodes:
            state = _run_node(node_name, node, state, index)
            if state.status == "failed":
                break

    state.metrics["workflow_duration_ms"] = (time.perf_counter() - workflow_started) * 1000
    trace_path = write_trace(state, trace_dir) if trace_dir else None
    return AgentResult(state=state, trace_path=trace_path)


def _run_node(node_name: str, node: AgentNode, state: AgentState, index: RagIndex) -> AgentState:
    started_at = utc_now_iso()
    start_time = time.perf_counter()
    input_summary = summarize_state(state)
    error = None
    status = "completed"

    try:
        state = node(state, index)
    except Exception as exc:  # pragma: no cover - defensive trace path
        status = "failed"
        error = str(exc)
        state.status = "failed"
        state.verification_notes.append(f"Node {node_name} failed: {exc}")

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
