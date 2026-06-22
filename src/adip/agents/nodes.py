"""Workflow nodes for the baseline document intelligence agent."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from adip.agents.models import AgentState
from adip.llmops.pipeline import generate_grounded_response
from adip.rag.retriever import RagIndex

BRIEF_KEYWORDS = {
    "brief",
    "report",
    "summarize",
    "summary",
    "overview",
    "research brief",
    "write up",
}

DOMAIN_FOCUS = {
    "academic": ["problem", "method", "dataset", "result", "limitation"],
    "finance": ["market", "metric", "risk", "growth", "revenue"],
    "crypto": ["protocol", "token", "security", "governance", "consensus"],
    "legal": ["party", "obligation", "deadline", "clause", "compliance"],
    "technical": ["system", "architecture", "dependency", "deployment", "interface"],
    "general": ["claim", "evidence", "risk", "open question"],
}


def route_intent(state: AgentState, index: RagIndex) -> AgentState:
    del index
    requested = state.requested_task.lower()
    question_text = state.question.lower()

    if requested in {"qa", "brief"}:
        state.task_type = requested
    elif any(keyword in question_text for keyword in BRIEF_KEYWORDS):
        state.task_type = "brief"
    else:
        state.task_type = "qa"

    state.status = "routed"
    state.metrics["requested_task"] = requested
    state.metrics["resolved_task_type"] = state.task_type
    return state


def plan_task(state: AgentState, index: RagIndex) -> AgentState:
    del index
    if state.task_type == "brief":
        focus = DOMAIN_FOCUS.get(state.domain_preset, DOMAIN_FOCUS["general"])
        state.plan = [
            "Retrieve the most relevant document chunks.",
            "Check whether retrieved evidence is sufficient.",
            f"Organize findings for the {state.domain_preset} domain.",
            f"Highlight focus areas: {', '.join(focus)}.",
            "Write a grounded research brief with citations.",
        ]
    else:
        state.plan = [
            "Retrieve the most relevant document chunks.",
            "Check whether the retrieved evidence supports an answer.",
            "Return a concise cited answer.",
        ]

    state.metrics["reasoning_planner_enabled"] = bool(
        state.use_reasoning_planner
        and (state.reasoning_provider or state.reasoning_model_name or state.reasoning_model_profile)
    )
    if state.metrics["reasoning_planner_enabled"]:
        try:
            result = generate_grounded_response(
                question=state.question,
                task_type="plan",
                domain_preset=state.domain_preset,
                retrieved=[],
                provider=state.reasoning_provider,
                model_name=state.reasoning_model_name,
                model_profile_id=state.reasoning_model_profile,
                model_profiles_path=Path(state.model_profiles_path),
                endpoint_url=state.reasoning_endpoint_url or state.endpoint_url,
                api_key=state.reasoning_api_key or state.api_key,
                device=state.reasoning_device or state.device,
                prompt_dir=Path(state.prompt_dir),
                prompt_version=state.reasoning_prompt_version,
                max_new_tokens=state.reasoning_max_new_tokens,
            )
            state.planning_llmops = result.metadata()
            state.planning_notes.append(result.answer)
            state.metrics.update(prefix_metrics(result.metrics(), prefix="planning_"))
        except Exception as exc:  # pragma: no cover - exercised through agent fallback tests
            state.planning_notes.append(
                f"Reasoning planner skipped because the optional reasoning model failed: {exc}"
            )
            state.metrics["planning_llm_error"] = str(exc)

    state.status = "planned"
    state.metrics["plan_step_count"] = len(state.plan)
    return state


def retrieve_evidence(state: AgentState, index: RagIndex) -> AgentState:
    retrieved = index.search(
        state.question,
        top_k=state.top_k,
        document_filter=state.document_filter,
    )
    state.retrieved = [item.to_dict() for item in retrieved]
    state.citations = [item.citation_label for item in retrieved]
    state.status = "retrieved"
    state.metrics["retrieved_count"] = len(retrieved)
    state.metrics["max_retrieval_score"] = max((item.score for item in retrieved), default=0.0)
    state.metrics["document_filter"] = state.document_filter
    return state


def verify_evidence(state: AgentState, index: RagIndex) -> AgentState:
    del index
    notes: list[str] = []
    if not state.retrieved:
        notes.append("No relevant chunks were retrieved. The answer should refuse or ask for more documents.")
    else:
        notes.append(f"Retrieved {len(state.retrieved)} candidate evidence chunks.")

    unique_documents = {item["chunk"]["document_id"] for item in state.retrieved}
    if unique_documents:
        notes.append(f"Evidence spans {len(unique_documents)} document(s).")

    low_score_count = sum(1 for item in state.retrieved if item["score"] < 0.05)
    if low_score_count:
        notes.append(f"{low_score_count} retrieved chunk(s) have low lexical similarity.")

    state.verification_notes = notes
    state.status = "verified"
    state.metrics["unique_document_count"] = len(unique_documents)
    state.metrics["low_score_count"] = low_score_count
    state.metrics["reasoning_verifier_enabled"] = bool(
        state.reasoning_provider or state.reasoning_model_name or state.reasoning_model_profile
    )
    if state.metrics["reasoning_verifier_enabled"]:
        try:
            result = generate_grounded_response(
                question=state.question,
                task_type="verify",
                domain_preset=state.domain_preset,
                retrieved=state.retrieved,
                provider=state.reasoning_provider,
                model_name=state.reasoning_model_name,
                model_profile_id=state.reasoning_model_profile,
                model_profiles_path=Path(state.model_profiles_path),
                endpoint_url=state.reasoning_endpoint_url or state.endpoint_url,
                api_key=state.reasoning_api_key or state.api_key,
                device=state.reasoning_device or state.device,
                prompt_dir=Path(state.prompt_dir),
                prompt_version=state.reasoning_prompt_version,
                max_new_tokens=state.reasoning_max_new_tokens,
            )
            state.reasoning_llmops = result.metadata()
            verifier_text = result.answer
            structured_output = result.structured_output or {}
            if structured_output:
                verifier_text = structured_output["final_text"]
            state.verification_notes.append(f"Reasoning verifier output:\n{verifier_text}")
            state.metrics.update(prefix_metrics(result.metrics(), prefix="reasoning_"))
        except Exception as exc:  # pragma: no cover - exercised through agent fallback tests
            state.verification_notes.append(
                f"Reasoning verifier skipped because the optional reasoning model failed: {exc}"
            )
            state.metrics["reasoning_llm_error"] = str(exc)
    return state


def write_response(state: AgentState, index: RagIndex) -> AgentState:
    del index
    result = generate_grounded_response(
        question=state.question,
        task_type=state.task_type,
        domain_preset=state.domain_preset,
        retrieved=state.retrieved,
        provider=state.llm_provider,
        model_name=state.model_name,
        model_profile_id=state.model_profile,
        model_profiles_path=Path(state.model_profiles_path),
        endpoint_url=state.endpoint_url,
        api_key=state.api_key,
        device=state.device,
        prompt_dir=Path(state.prompt_dir),
        prompt_version=state.prompt_version,
        max_new_tokens=state.max_new_tokens,
        reasoning_effort=state.reasoning_effort,
    )
    state.final_answer = result.answer
    state.llmops = result.metadata()
    state.metrics.update(result.metrics())

    state.status = "written"
    state.metrics["answer_char_count"] = len(state.final_answer)
    return state


def check_citations(state: AgentState, index: RagIndex) -> AgentState:
    del index
    missing = [citation for citation in state.citations if citation not in state.final_answer]
    if not state.citations:
        state.verification_notes.append("No citations are available for the final answer.")
    elif missing:
        state.verification_notes.append(f"{len(missing)} citation(s) were retrieved but not shown in the final answer.")
    else:
        state.verification_notes.append("All retrieved citations are visible in the final answer.")

    state.status = "completed"
    state.metrics["citation_count"] = len(state.citations)
    state.metrics["visible_citation_count"] = len(state.citations) - len(missing)
    return state


def build_research_brief(state: AgentState) -> str:
    if not state.retrieved:
        return (
            f"Research Brief: {state.question}\n\n"
            "Status: insufficient evidence.\n\n"
            "I could not find relevant evidence in the indexed documents."
        )

    top_chunks = state.retrieved[: min(5, len(state.retrieved))]
    domain_focus = DOMAIN_FOCUS.get(state.domain_preset, DOMAIN_FOCUS["general"])
    source_counts = Counter(item["chunk"]["filename"] for item in top_chunks)

    evidence_lines = []
    for index, item in enumerate(top_chunks, start=1):
        chunk = item["chunk"]
        citation = item["citation"]
        snippet = " ".join(chunk["text"].split())
        if len(snippet) > 420:
            snippet = f"{snippet[:417].rstrip()}..."
        evidence_lines.append(f"{index}. {snippet} ({citation})")

    source_summary = ", ".join(f"{name}: {count}" for name, count in source_counts.items())
    verification = "\n".join(f"- {note}" for note in state.verification_notes)

    return (
        f"Research Brief: {state.question}\n\n"
        f"Domain preset: {state.domain_preset}\n"
        f"Task type: {state.task_type}\n"
        f"Focus areas: {', '.join(domain_focus)}\n\n"
        "Evidence Summary:\n"
        f"{chr(10).join(evidence_lines)}\n\n"
        "Source Coverage:\n"
        f"{source_summary}\n\n"
        "Verification Notes:\n"
        f"{verification}\n\n"
        "Current Limitation:\n"
        "This brief is an extractive baseline. LLM synthesis will be added after the workflow is tracked and testable."
    )


def _retrieved_records_to_items(retrieved: list[dict[str, Any]]):
    from adip.rag.retriever import RetrievedChunk

    return [
        RetrievedChunk(chunk=item["chunk"], score=float(item["score"]), rank=int(item["rank"]))
        for item in retrieved
    ]


def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}
