"""Workflow nodes for the baseline document intelligence agent."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

from adip.agents.models import AgentState
from adip.llmops.pipeline import generate_grounded_response
from adip.rag.coverage import (
    FOLLOW_UP_ACCEPT_COVERAGE,
    assess_coverage,
    content_terms,
    term_statistics_for,
    weighted_coverage,
)
from adip.rag.decompose import parse_sub_questions, split_question
from adip.rag.retriever import RagIndex

SUPPORTED_DECOMPOSERS = {"none", "rules", "llm"}

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

    state.sub_questions = decompose_question(state)
    state.pending_queries = initial_queries(state)
    if len(state.sub_questions) > 1:
        state.plan.insert(
            0,
            f"Split the question into {len(state.sub_questions)} sub-questions and retrieve evidence for each.",
        )
    state.status = "planned"
    state.metrics["plan_step_count"] = len(state.plan)
    state.metrics["sub_question_count"] = len(state.sub_questions)
    state.metrics["decomposer"] = state.decomposer
    return state


def decompose_question(state: AgentState) -> list[str]:
    """Sub-questions for the retrieval loop (``[question]`` when it is not split)."""
    if state.decomposer not in SUPPORTED_DECOMPOSERS:
        raise ValueError(f"Unsupported decomposer: {state.decomposer}")
    if state.decomposer == "none":
        return [state.question]
    if state.decomposer == "llm":
        if not (state.reasoning_provider or state.reasoning_model_name or state.reasoning_model_profile):
            state.planning_notes.append("LLM decomposition needs a reasoning model; used rule-based splitting.")
        else:
            try:
                result = generate_grounded_response(
                    question=state.question,
                    task_type="decompose",
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
                    max_new_tokens=state.reasoning_max_new_tokens,
                )
                state.metrics.update(prefix_metrics(result.metrics(), prefix="decompose_"))
                return parse_sub_questions(result.answer, state.question)
            except Exception as exc:  # the planner model is optional: fall back, never fail the run
                state.planning_notes.append(f"LLM decomposition failed ({exc}); used rule-based splitting.")
                state.metrics["decompose_llm_error"] = str(exc)
    return split_question(state.question)


def initial_queries(state: AgentState) -> list[dict[str, Any]]:
    """First retrieval round: the question itself, plus one search per sub-question when it was split.

    Keeping the whole question as the first stream means decomposition only ever adds
    evidence: the single-pass top result still leads the evidence list.
    """
    if len(state.sub_questions) <= 1:
        return [{"query": state.question, "sub_question_index": 0, "reason": "question"}]
    queries: list[dict[str, Any]] = [{"query": state.question, "sub_question_index": None, "reason": "question"}]
    queries.extend(
        {"query": sub_question, "sub_question_index": index, "reason": "sub_question"}
        for index, sub_question in enumerate(state.sub_questions)
    )
    return queries


def retrieve_evidence(state: AgentState, index: RagIndex) -> AgentState:
    """Run every pending search, then rebuild the evidence list from all accepted streams."""
    started = time.perf_counter()
    queries = state.pending_queries or [{"query": state.question, "sub_question_index": 0, "reason": "question"}]
    state.pending_queries = []
    round_number = int(state.metrics.get("retrieval_round_count", 0)) + 1
    stats = term_statistics_for(index) if any(q["reason"] == "follow_up" for q in queries) else None

    for query in queries:
        found = index.search(query["query"], top_k=state.top_k, document_filter=state.document_filter)
        results = [item.to_dict() for item in found]
        accepted = True
        if query["reason"] == "follow_up":
            # Keep a follow-up search only if it found passages about the missing terms,
            # best match first; otherwise it would just add noise to the evidence.
            missing = set(query.get("follow_up_terms") or [])
            scored = [
                (weighted_coverage(missing, content_terms(item["chunk"]["text"]), stats), item) for item in results
            ]
            scored.sort(key=lambda pair: -pair[0])
            accepted = bool(scored) and scored[0][0] >= FOLLOW_UP_ACCEPT_COVERAGE
            results = [item for _, item in scored]
        stream = {
            "stream": len(state.retrieval_streams),
            "round": round_number,
            "query": query["query"],
            "reason": query["reason"],
            "sub_question_index": query.get("sub_question_index"),
            "accepted": accepted,
            "chunk_ids": [str(item["chunk"]["chunk_id"]) for item in results],
            "top_score": results[0]["score"] if results else 0.0,
        }
        state.retrieval_streams.append(stream)
        if accepted:
            for item in results:
                state.evidence_pool.setdefault(str(item["chunk"]["chunk_id"]), item)

    state.retrieved = assemble_evidence(state)
    state.citations = [item["citation"] for item in state.retrieved]
    state.retrieval_elapsed_ms += (time.perf_counter() - started) * 1000
    state.status = "retrieved"
    state.metrics["retrieved_count"] = len(state.retrieved)
    state.metrics["max_retrieval_score"] = max((item["score"] for item in state.retrieved), default=0.0)
    state.metrics["document_filter"] = state.document_filter
    state.metrics["retrieval_round_count"] = round_number
    state.metrics["retrieval_call_count"] = int(state.metrics.get("retrieval_call_count", 0)) + len(queries)
    state.metrics["retrieval_elapsed_ms"] = state.retrieval_elapsed_ms
    return state


def assemble_evidence(state: AgentState) -> list[dict[str, Any]]:
    """Interleave accepted streams round-robin (their best results first), dedupe, cap at top_k.

    With a single stream this is exactly that search's top_k results, so a question the
    loop never splits or retries gets the same evidence as a plain search.
    """
    streams = [stream for stream in state.retrieval_streams if stream["accepted"]]
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    depth = 0
    while len(evidence) < state.top_k and any(depth < len(stream["chunk_ids"]) for stream in streams):
        for stream in streams:
            if depth >= len(stream["chunk_ids"]) or len(evidence) >= state.top_k:
                continue
            chunk_id = stream["chunk_ids"][depth]
            if chunk_id in seen or chunk_id not in state.evidence_pool:
                continue
            seen.add(chunk_id)
            item = dict(state.evidence_pool[chunk_id])
            item["rank"] = len(evidence) + 1
            item["query"] = stream["query"]
            item["sub_question_index"] = stream["sub_question_index"]
            item["retrieval_reason"] = stream["reason"]
            evidence.append(item)
        depth += 1
    return evidence


def verify_evidence(state: AgentState, index: RagIndex) -> AgentState:
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
    schedule_follow_up_searches(state, index)
    state.metrics["retrieval_retry_count"] = len(state.retrieval_retries)
    state.metrics["reasoning_verifier_enabled"] = bool(
        state.reasoning_provider or state.reasoning_model_name or state.reasoning_model_profile
    )
    if state.pending_queries:
        # Another retrieval round is coming; judge the evidence once it is final.
        return state
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


def schedule_follow_up_searches(state: AgentState, index: RagIndex) -> None:
    """Check each (sub-)question's coverage by the evidence and queue searches for what is missing.

    A (sub-)question is supported when its best passage, together with passages that earlier
    follow-up searches found for it, covers at least ``coverage_threshold`` of its IDF-weighted
    content terms. Otherwise its missing *distinctive* terms become the next query. The loop
    stops when everything is supported, nothing new is left to search for, or a budget runs out.
    """
    if not state.retrieved:
        state.sub_question_support = []
        return
    stats = term_statistics_for(index)
    sub_questions = state.sub_questions or [state.question]
    issued = {(stream["sub_question_index"], stream["query"]) for stream in state.retrieval_streams}
    support: list[dict[str, Any]] = []
    follow_ups: list[dict[str, Any]] = []
    for position, sub_question in enumerate(sub_questions):
        found_by_follow_up = [
            item["chunk"]
            for item in state.retrieved
            if item.get("retrieval_reason") == "follow_up" and item.get("sub_question_index") == position
        ]
        result = assess_coverage(
            sub_question,
            [item["chunk"] for item in state.retrieved],
            stats,
            extra_chunks=found_by_follow_up,
        )
        supported = result.coverage >= state.coverage_threshold
        support.append({"index": position, "sub_question": sub_question, "supported": supported, **result.to_dict()})
        query = " ".join(result.follow_up_terms)
        if not supported and query and (position, query) not in issued:
            label = "question" if len(sub_questions) == 1 else f"sub-question {position + 1}"
            follow_ups.append(
                {
                    "query": query,
                    "sub_question_index": position,
                    "reason": "follow_up",
                    "follow_up_terms": result.follow_up_terms,
                    "retry_reason": (
                        f"{label} coverage {result.coverage:.2f} below {state.coverage_threshold:.2f}, "
                        f"missing {', '.join(result.specific_missing_terms[:5])}"
                    ),
                }
            )
    state.sub_question_support = support
    state.metrics["min_evidence_coverage"] = min(entry["coverage"] for entry in support)
    state.metrics["unsupported_sub_question_count"] = sum(1 for entry in support if not entry["supported"])

    for stream in state.retrieval_streams:
        if stream["reason"] == "follow_up":
            outcome = "found new evidence" if stream["accepted"] else "found nothing relevant, discarded"
            state.verification_notes.append(f"Round {stream['round']} follow-up search '{stream['query']}': {outcome}.")
    for entry in support:
        label = "Question" if len(support) == 1 else f"Sub-question {entry['index'] + 1}"
        status = "covered" if entry["supported"] else "not fully covered"
        missing = f" (missing: {', '.join(entry['missing_terms'][:5])})" if entry["missing_terms"] else ""
        state.verification_notes.append(f"{label} {status}: coverage {entry['coverage']:.2f}{missing}.")
    if not follow_ups:
        return

    retries_used = int(state.metrics.get("retrieval_round_count", 1)) - 1
    calls_left = state.max_retrieval_calls - int(state.metrics.get("retrieval_call_count", 0))
    if retries_used >= state.max_retrieval_retries:
        stop_reason = f"retry limit reached ({state.max_retrieval_retries})"
    elif calls_left <= 0:
        stop_reason = f"search budget used up ({state.max_retrieval_calls} searches)"
    elif state.retrieval_elapsed_ms >= state.retrieval_time_budget_ms:
        stop_reason = f"retrieval time budget used up ({state.retrieval_time_budget_ms:.0f} ms)"
    else:
        state.pending_queries = follow_ups[:calls_left]
        state.retrieval_retries.append(
            {
                "retry": len(state.retrieval_retries) + 1,
                "round": retries_used + 2,
                "searches": [
                    {
                        "query": follow_up["query"],
                        "sub_question_index": follow_up["sub_question_index"],
                        "reason": follow_up["retry_reason"],
                    }
                    for follow_up in state.pending_queries
                ],
            }
        )
        for follow_up in state.pending_queries:
            state.verification_notes.append(f"Searching again for: {follow_up['query']} ({follow_up['retry_reason']}).")
        return
    state.metrics["retrieval_stop_reason"] = stop_reason
    state.verification_notes.append(f"Not searching again: {stop_reason}.")


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
