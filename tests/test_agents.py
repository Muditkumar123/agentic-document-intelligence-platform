from adip.agents.runner import run_agent
from adip.rag.retriever import build_index


def make_chunk(chunk_id, text):
    return {
        "chunk_id": chunk_id,
        "document_id": "doc_test",
        "filename": "sample.md",
        "source_path": "/tmp/sample.md",
        "source_type": "md",
        "checksum": "abc123",
        "page_number": 1,
        "chunk_index": 0,
        "text": text,
        "token_count": len(text.split()),
        "char_count": len(text),
        "metadata": {},
    }


def test_agent_runs_qa_workflow_with_trace(tmp_path):
    index = build_index(
        [
            make_chunk("chunk_platform", "The platform ingests documents and writes JSONL chunks."),
            make_chunk("chunk_ops", "AgentOps records node traces and tool calls."),
        ]
    )

    result = run_agent(
        question="What does the platform ingest?",
        index=index,
        task="qa",
        top_k=2,
        model_profile="extractive_baseline",
        trace_dir=tmp_path,
    )

    assert result.state.status == "completed"
    assert result.state.task_type == "qa"
    assert "platform ingests documents" in result.state.final_answer
    assert result.state.llmops["prompt"]["version"] == "qa_v1"
    assert result.state.llmops["model_profile"]["profile_id"] == "extractive_baseline"
    assert result.state.metrics["llm_input_token_count"] > 0
    assert len(result.state.trace) == 6
    assert [event.node_name for event in result.state.trace] == [
        "intent_router",
        "planner",
        "retriever",
        "evidence_verifier",
        "writer",
        "citation_checker",
    ]
    assert result.trace_path is not None


def test_agent_routes_and_writes_brief():
    index = build_index(
        [
            make_chunk(
                "chunk_brief",
                "The system creates research briefs from retrieved document evidence with citations.",
            ),
            make_chunk("chunk_other", "Monitoring tracks latency and failure cases."),
        ]
    )

    result = run_agent(
        question="Create a research brief about document evidence.",
        index=index,
        task="auto",
        domain_preset="academic",
        top_k=2,
    )

    assert result.state.status == "completed"
    assert result.state.task_type == "brief"
    assert "Research Brief" in result.state.final_answer
    assert "Domain preset: academic" in result.state.final_answer
    assert result.state.llmops["prompt"]["version"] == "brief_v1"
    assert result.state.metrics["citation_count"] >= 1


def test_agent_document_filter_targets_selected_document():
    index = build_index(
        [
            make_chunk("chunk_simon", "SIMON paper evidence for neural distinguishers."),
            {
                **make_chunk("chunk_notes", "SIMON appears in unrelated platform notes."),
                "document_id": "doc_notes",
                "filename": "notes.md",
            },
        ]
    )

    result = run_agent(
        question="summarize this document",
        index=index,
        task="brief",
        top_k=1,
        document_filter="notes.md",
        model_profile="extractive_baseline",
    )

    assert result.state.status == "completed"
    assert result.state.retrieved[0]["chunk"]["filename"] == "notes.md"
    assert result.state.metrics["document_filter"] == "notes.md"


def test_agent_can_run_reasoning_verifier():
    index = build_index(
        [
            make_chunk(
                "chunk_reasoning",
                "The verifier checks retrieved evidence before the writer creates the final answer.",
            )
        ]
    )

    result = run_agent(
        question="What does the verifier check?",
        index=index,
        task="qa",
        top_k=1,
        model_profile="extractive_baseline",
        reasoning_model_profile="extractive_baseline",
        reasoning_max_new_tokens=128,
    )

    assert result.state.status == "completed"
    assert result.state.reasoning_llmops["prompt"]["version"] == "verify_v1"
    assert result.state.reasoning_llmops["structured_output"]["structured"] is True
    assert result.state.metrics["reasoning_verifier_enabled"] is True
    assert result.state.metrics["reasoning_llm_structured_output"] == 1.0
    assert result.state.metrics["reasoning_llm_input_token_count"] > 0
    assert any("Reasoning verifier output" in note for note in result.state.verification_notes)


def test_agent_can_run_reasoning_planner_and_verifier():
    index = build_index(
        [
            make_chunk(
                "chunk_plan",
                "The planner decides what to retrieve and the verifier checks evidence before writing.",
            )
        ]
    )

    result = run_agent(
        question="Plan and answer a workflow question.",
        index=index,
        task="qa",
        top_k=1,
        model_profile="extractive_baseline",
        reasoning_model_profile="extractive_baseline",
        reasoning_max_new_tokens=128,
        use_reasoning_planner=True,
    )

    assert result.state.status == "completed"
    assert result.state.planning_llmops["prompt"]["version"] == "plan_v1"
    assert result.state.reasoning_llmops["prompt"]["version"] == "verify_v1"
    assert result.state.metrics["reasoning_planner_enabled"] is True
    assert result.state.metrics["planning_llm_input_token_count"] > 0


def test_agent_continues_when_optional_reasoning_verifier_fails(monkeypatch):
    import adip.agents.nodes as agent_nodes

    original_generate = agent_nodes.generate_grounded_response

    def fail_for_reasoning(*args, **kwargs):
        if kwargs.get("task_type") == "verify":
            raise RuntimeError("HTTP Error 402: Payment Required")
        return original_generate(*args, **kwargs)

    monkeypatch.setattr(agent_nodes, "generate_grounded_response", fail_for_reasoning)
    index = build_index(
        [
            make_chunk(
                "chunk_verifier_fallback",
                "The writer can still produce cited answers when optional reasoning verification is unavailable.",
            )
        ]
    )

    result = run_agent(
        question="What can the writer still produce?",
        index=index,
        task="qa",
        top_k=1,
        model_profile="extractive_baseline",
        reasoning_model_profile="deepseek_v4_pro_cloud",
    )

    assert result.state.status == "completed"
    assert "cited answers" in result.state.final_answer
    assert result.state.metrics["reasoning_llm_error"] == "HTTP Error 402: Payment Required"
    assert any("Reasoning verifier skipped" in note for note in result.state.verification_notes)


def test_agent_continues_when_optional_reasoning_planner_fails(monkeypatch):
    import adip.agents.nodes as agent_nodes

    original_generate = agent_nodes.generate_grounded_response

    def fail_for_planning(*args, **kwargs):
        if kwargs.get("task_type") == "plan":
            raise RuntimeError("HTTP Error 402: Payment Required")
        return original_generate(*args, **kwargs)

    monkeypatch.setattr(agent_nodes, "generate_grounded_response", fail_for_planning)
    index = build_index(
        [
            make_chunk(
                "chunk_planner_fallback",
                "The default planner retrieves evidence and writes a cited answer.",
            )
        ]
    )

    result = run_agent(
        question="What does the default planner do?",
        index=index,
        task="qa",
        top_k=1,
        model_profile="extractive_baseline",
        reasoning_model_profile="deepseek_v4_pro_cloud",
        use_reasoning_planner=True,
    )

    assert result.state.status == "completed"
    assert "retrieves evidence" in result.state.final_answer
    assert result.state.metrics["planning_llm_error"] == "HTTP Error 402: Payment Required"
    assert any("Reasoning planner skipped" in note for note in result.state.planning_notes)
