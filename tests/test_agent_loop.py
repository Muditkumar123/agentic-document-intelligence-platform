"""Retrieval loop: question decomposition, coverage-driven follow-up searches, and routing."""

import json

import pytest

from adip.agents.graph import langgraph_available
from adip.agents.multihop_eval import (
    aggregate_multihop,
    answer_covers_hop,
    hop_supported,
    read_probes,
    score_multihop_case,
)
from adip.agents.runner import run_agent
from adip.rag.coverage import assess_coverage, build_term_statistics, content_terms, entity_terms
from adip.rag.decompose import parse_sub_questions, split_question
from adip.rag.retriever import build_index

ENGINES = ["sequential"] + (["langgraph"] if langgraph_available() else [])


def make_chunk(chunk_id, text, filename="sample.md"):
    return {
        "chunk_id": chunk_id,
        "document_id": f"doc_{filename}",
        "filename": filename,
        "source_path": f"/tmp/{filename}",
        "source_type": "md",
        "checksum": "abc123",
        "page_number": 1,
        "chunk_index": 0,
        "text": text,
        "token_count": len(text.split()),
        "char_count": len(text),
        "metadata": {},
    }


# --- decomposition -----------------------------------------------------------------


def test_split_question_splits_coordinated_questions():
    assert split_question("Which protocol does OAuth require, and how does TLS 1.3 send early data?") == [
        "Which protocol does OAuth require",
        "How does TLS 1.3 send early data?",
    ]
    assert split_question("Is POST idempotent? Also, what is the maximum length of a DNS label?") == [
        "Is POST idempotent?",
        "What is the maximum length of a DNS label?",
    ]


def test_split_question_leaves_single_questions_and_noun_phrases_whole():
    for question in [
        "What are the principles for processing personal data under the GDPR?",
        "Which methods are safe and idempotent in HTTP?",
        "How do the Transformer and BERT differ?",
        "What does the U.S. Securities and Exchange Commission require in a Form 10-K?",
    ]:
        assert split_question(question) == [question]


def test_split_question_carries_a_colon_context_onto_each_part():
    parts = split_question("Compare GDPR with the AI Act: what does each require, and which one bans social scoring?")

    assert parts == [
        "Compare GDPR with the AI Act: What does each require",
        "Compare GDPR with the AI Act: Which one bans social scoring?",
    ]


def test_split_question_rejoins_fragments_and_caps_parts():
    assert split_question("What is TLS? Why?") == ["What is TLS? Why?"]
    many = " ".join(f"What is topic number {n}?" for n in range(6))
    assert len(split_question(many)) == 4


def test_parse_sub_questions_reads_numbered_model_output():
    text = "Here you go:\n1. Which protocol does OAuth 2.0 require?\n2) How does TLS 1.3 send early data?\n"

    assert parse_sub_questions(text, "q") == [
        "Which protocol does OAuth 2.0 require?",
        "How does TLS 1.3 send early data?",
    ]
    assert parse_sub_questions("1. What is OAuth used for?", "What is OAuth used for?") == ["What is OAuth used for?"]


# --- coverage ----------------------------------------------------------------------


def test_content_and_entity_terms():
    assert content_terms("What are the refresh tokens used for?") == {"refresh", "token"}
    assert entity_terms("Is POST idempotent? Also, what limits a DNS label in OAuth?") == {"post", "dns", "oauth"}
    assert entity_terms("Which protocol is used?") == set()


def test_assess_coverage_flags_a_missing_named_entity():
    chunks = [
        {"chunk_id": "a", "text": "The OAuth authorization server issues access tokens to the client."},
        {"chunk_id": "b", "text": "Each DNS label must be 63 octets or less."},
        {"chunk_id": "c", "text": "OAuth clients present access tokens to the resource server."},
    ]
    stats = build_term_statistics(chunk["text"] for chunk in chunks)

    covered = assess_coverage("Who issues OAuth access tokens?", chunks[:1], stats)
    missing = assess_coverage("Who issues OAuth access tokens, and how long is a DNS label?", [chunks[0]], stats)

    assert covered.coverage == pytest.approx(1.0)
    assert covered.follow_up_terms == []
    assert missing.coverage < 0.8
    assert missing.follow_up_terms[0] == "dns"
    assert {"dns", "label"} <= set(missing.follow_up_terms)


# --- the loop ----------------------------------------------------------------------

MIXED_CHUNKS = [
    make_chunk("oauth_issue", "The OAuth authorization server issues access tokens to the client after authentication.", "oauth.md"),
    make_chunk("oauth_present", "OAuth clients present access tokens from the authorization server to the resource server.", "oauth.md"),
    make_chunk("dns_label", "Each DNS label must be 63 octets or less, and a full domain name 255 octets or less.", "dns.md"),
    make_chunk("tls_flight", "TLS 1.3 lets clients send early data in the first flight using a pre-shared key.", "tls.md"),
]


@pytest.mark.parametrize("engine", ENGINES)
def test_single_hop_question_takes_one_retrieval_round(engine):
    index = build_index(MIXED_CHUNKS)
    question = "Who issues OAuth access tokens to the client?"

    state = run_agent(question, index, task="qa", top_k=2, model_profile="extractive_baseline", engine=engine).state

    assert state.status == "completed"
    assert [event.node_name for event in state.trace] == [
        "intent_router", "planner", "retriever", "evidence_verifier", "writer", "citation_checker",
    ]
    assert state.metrics["retrieval_round_count"] == 1
    assert state.metrics["retrieval_call_count"] == 1
    assert [item["chunk"]["chunk_id"] for item in state.retrieved] == [
        item.chunk["chunk_id"] for item in index.search(question, top_k=2)
    ]


@pytest.mark.parametrize("engine", ENGINES)
def test_decomposed_question_retrieves_evidence_for_each_part(engine):
    index = build_index(MIXED_CHUNKS)
    question = "Who issues OAuth access tokens, and how long may a DNS label be?"

    split = run_agent(question, index, task="qa", top_k=2, model_profile="extractive_baseline", engine=engine).state

    assert split.sub_questions == ["Who issues OAuth access tokens", "How long may a DNS label be?"]
    assert [stream["reason"] for stream in split.retrieval_streams] == ["question", "sub_question", "sub_question"]
    assert {item["chunk"]["filename"] for item in split.retrieved} == {"oauth.md", "dns.md"}
    # Each evidence item records which search found it, so the writer can focus on that part.
    assert {item["query"] for item in split.retrieved} <= {question, *split.sub_questions}
    assert split.metrics["retrieval_call_count"] == 3
    assert "63 octets" in split.final_answer


@pytest.mark.parametrize("engine", ENGINES)
def test_verifier_sends_the_run_back_to_retrieval_for_missing_terms(engine):
    index = build_index(MIXED_CHUNKS)
    question = "Which OAuth server issues access tokens to the client with DNS label octets"

    state = run_agent(
        question, index, task="qa", top_k=2, model_profile="extractive_baseline", engine=engine,
        decomposer="none", coverage_threshold=0.95,
    ).state

    names = [event.node_name for event in state.trace]
    assert names.count("retriever") == 2 and names.count("evidence_verifier") == 2
    follow_up = state.retrieval_streams[1]
    assert follow_up["reason"] == "follow_up" and follow_up["accepted"]
    assert "dns" in follow_up["query"]
    assert "dns_label" in [item["chunk"]["chunk_id"] for item in state.retrieved]
    assert state.metrics["retrieval_round_count"] == 2
    assert any("Round 2 follow-up search" in note and "found new evidence" in note for note in state.verification_notes)
    assert state.status == "completed"


def test_retry_limit_and_search_budget_stop_the_loop():
    index = build_index(MIXED_CHUNKS)
    question = "Which OAuth server issues access tokens to the client with DNS label octets"
    common = dict(task="qa", top_k=2, model_profile="extractive_baseline", decomposer="none", coverage_threshold=0.95)

    no_retries = run_agent(question, index, max_retrieval_retries=0, **common).state
    no_budget = run_agent(question, index, max_retrieval_calls=1, **common).state

    assert no_retries.metrics["retrieval_round_count"] == 1
    assert no_retries.metrics["retrieval_stop_reason"] == "retry limit reached (0)"
    assert no_budget.metrics["retrieval_round_count"] == 1
    assert "search budget" in no_budget.metrics["retrieval_stop_reason"]


def test_follow_up_that_finds_nothing_relevant_is_discarded():
    chunks = MIXED_CHUNKS + [make_chunk("filler", "Unrelated filler text about weather and gardening.", "misc.md")]
    index = build_index(chunks)

    state = run_agent(
        "Which OAuth server issues tokens and what is Kerberos", index, task="qa", top_k=2,
        model_profile="extractive_baseline", decomposer="none", coverage_threshold=0.95,
    ).state

    follow_ups = [stream for stream in state.retrieval_streams if stream["reason"] == "follow_up"]
    assert follow_ups and not any(stream["accepted"] for stream in follow_ups)
    assert all(item["retrieval_reason"] != "follow_up" for item in state.retrieved)


def test_llm_decomposer_uses_model_sub_questions_and_falls_back_on_failure(monkeypatch):
    import adip.agents.nodes as nodes_module

    index = build_index(MIXED_CHUNKS)
    real_generate = nodes_module.generate_grounded_response

    class FakeResult:
        answer = "1. Who issues OAuth access tokens?\n2. How long can a DNS label be?"

        def metrics(self):
            return {"llm_latency_ms": 1.0}

    def fake_generate(**kwargs):
        if kwargs["task_type"] == "decompose":
            return FakeResult()
        return real_generate(**kwargs)

    monkeypatch.setattr(nodes_module, "generate_grounded_response", fake_generate)
    question = "Tell me about OAuth token issuance plus DNS label size limits"
    state = run_agent(
        question, index, task="qa", top_k=2, model_profile="extractive_baseline", decomposer="llm",
        reasoning_provider="openai_compatible", reasoning_model_name="fake-model",
    ).state
    assert state.sub_questions == ["Who issues OAuth access tokens?", "How long can a DNS label be?"]

    def failing_generate(**kwargs):
        if kwargs["task_type"] == "decompose":
            raise RuntimeError("endpoint down")
        return real_generate(**kwargs)

    monkeypatch.setattr(nodes_module, "generate_grounded_response", failing_generate)
    state = run_agent(
        "Who issues OAuth access tokens, and how long may a DNS label be?", index, task="qa", top_k=2,
        model_profile="extractive_baseline", decomposer="llm", reasoning_provider="openai_compatible",
        reasoning_model_name="fake-model",
    ).state
    assert len(state.sub_questions) == 2  # rule-based fallback
    assert any("LLM decomposition failed" in note for note in state.planning_notes)


def test_both_engines_take_the_same_path_through_the_loop():
    if not langgraph_available():
        pytest.skip("langgraph not installed")
    index = build_index(MIXED_CHUNKS)
    kwargs = dict(
        question="Which OAuth server issues access tokens to the client with DNS label octets",
        index=index, task="qa", top_k=2, model_profile="extractive_baseline", decomposer="none",
        coverage_threshold=0.95,
    )

    sequential = run_agent(engine="sequential", **kwargs).state
    graphed = run_agent(engine="langgraph", **kwargs).state

    assert [e.node_name for e in graphed.trace] == [e.node_name for e in sequential.trace]
    assert graphed.retrieval_streams == sequential.retrieval_streams
    assert graphed.final_answer == sequential.final_answer


# --- multi-hop eval scoring --------------------------------------------------------


def test_multihop_scoring_and_probe_file(tmp_path):
    hop_dns = {"documents": ["dns.md"], "expected_substrings": ["63 octets"]}
    hop_oauth = {"documents": ["oauth.md"], "expected_substrings": ["issues access tokens"]}
    retrieved = [{"chunk": MIXED_CHUNKS[0]}, {"chunk": MIXED_CHUNKS[3]}]

    assert hop_supported(hop_oauth, retrieved)
    assert not hop_supported(hop_dns, retrieved)
    assert answer_covers_hop(hop_dns, "A label is limited to 63 octets.")

    probes_path = tmp_path / "probes.jsonl"
    probes_path.write_text(
        json.dumps({"id": "p1", "type": "comparison", "question": "q?", "hops": [hop_oauth, hop_dns]}) + "\n",
        encoding="utf-8",
    )
    probe = read_probes(probes_path)[0]
    index = build_index(MIXED_CHUNKS)
    state = run_agent(
        "Who issues OAuth access tokens, and how long may a DNS label be?", index, task="qa", top_k=2,
        model_profile="extractive_baseline",
    ).state
    summary = aggregate_multihop([score_multihop_case(probe, state)])

    assert summary["full_support_rate"] == 1.0
    assert summary["decomposition_rate"] == 1.0
    assert summary["avg_retrieval_calls"] == 3.0

    probes_path.write_text(json.dumps({"id": "bad", "question": "q?", "hops": [hop_oauth]}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least two hops"):
        read_probes(probes_path)
