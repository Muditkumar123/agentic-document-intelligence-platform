import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402
from opentelemetry.trace import StatusCode  # noqa: E402

import adip.tracing as tracing  # noqa: E402
from adip.agents.graph import langgraph_available  # noqa: E402
from adip.agents.runner import run_agent  # noqa: E402
from adip.rag.retriever import build_index  # noqa: E402

EXPORTER = InMemorySpanExporter()
QUESTION = "Who issues OAuth access tokens, and how long may a DNS label be?"


def make_chunk(chunk_id, text, filename):
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


CHUNKS = [
    make_chunk("oauth", "The authorization server issues OAuth access tokens to the client.", "rfc6749.md"),
    make_chunk("dns", "A DNS label may be at most 63 octets long.", "rfc1035.md"),
    make_chunk("tls", "TLS 1.3 removes renegotiation and static RSA key exchange.", "rfc8446.md"),
]


@pytest.fixture(scope="module", autouse=True)
def in_memory_provider():
    # OpenTelemetry accepts one global provider per process; this module installs it.
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
    trace.set_tracer_provider(provider)
    yield


@pytest.fixture(autouse=True)
def clear_spans():
    EXPORTER.clear()
    yield


@pytest.mark.parametrize("engine", ["sequential", "langgraph"])
def test_agent_run_is_one_span_tree(engine):
    if engine == "langgraph" and not langgraph_available():
        pytest.skip("langgraph not installed")

    result = run_agent(QUESTION, build_index(CHUNKS), task="qa", top_k=2, engine=engine, api_key="sk-test-secret")

    spans = EXPORTER.get_finished_spans()
    by_id = {span.context.span_id: span for span in spans}
    [root] = [span for span in spans if span.parent is None]
    assert root.name == "invoke_agent adip"
    assert root.attributes["adip.engine"] == engine
    assert root.attributes["adip.run_id"] == result.state.run_id
    assert root.attributes["adip.status"] == "completed"

    nodes = [span for span in spans if span.name.startswith("agent.node ")]
    assert [span.attributes["adip.node"] for span in sorted(nodes, key=lambda s: s.start_time)] == [
        event.node_name for event in result.state.trace
    ]
    assert all(span.parent.span_id == root.context.span_id for span in nodes)

    # Every search sits under a retriever node, and there is one per retrieval call.
    searches = [span for span in spans if span.name == "retrieval.search"]
    assert len(searches) == result.state.metrics["retrieval_call_count"]
    assert {by_id[span.parent.span_id].attributes["adip.node"] for span in searches} == {"retriever"}
    assert all(span.attributes["adip.retrieval.result_count"] >= 1 for span in searches)

    [chat] = [span for span in spans if span.name.startswith("chat ")]
    assert by_id[chat.parent.span_id].attributes["adip.node"] == "writer"
    assert chat.attributes["gen_ai.operation.name"] == "chat"
    assert chat.attributes["gen_ai.provider.name"] == "extractive"
    assert chat.attributes["gen_ai.usage.output_tokens"] > 0


def test_spans_never_carry_question_text_or_keys():
    run_agent(QUESTION, build_index(CHUNKS), task="qa", top_k=2, engine="sequential", api_key="sk-test-secret")

    values = [str(value) for span in EXPORTER.get_finished_spans() for value in span.attributes.values()]
    assert values
    assert not any("OAuth" in value or "sk-test-secret" in value for value in values)


def test_a_failing_block_marks_its_span_as_an_error():
    with pytest.raises(RuntimeError):
        with tracing.span("boom"):
            raise RuntimeError("model endpoint down")
    with tracing.span("handled") as current:
        tracing.mark_error(current, "node failed")

    boom, handled = EXPORTER.get_finished_spans()
    assert boom.status.status_code == StatusCode.ERROR
    assert handled.status.status_code == StatusCode.ERROR


def test_tracing_is_a_no_op_without_opentelemetry(monkeypatch):
    monkeypatch.setattr(tracing, "otel_trace", None)

    with tracing.span("ignored", {"adip.node": "x"}) as current:
        tracing.set_attributes(current, {"adip.status": "ok"})
        tracing.mark_error(current, "ignored")
    run_agent(QUESTION, build_index(CHUNKS), task="qa", top_k=2, engine="sequential")

    assert current is None
    assert EXPORTER.get_finished_spans() == ()


def test_exporter_is_installed_only_with_an_endpoint_and_no_provider(monkeypatch):
    installed = []
    monkeypatch.setattr(tracing.otel_trace, "get_tracer_provider", lambda: trace.ProxyTracerProvider())
    monkeypatch.setattr(tracing.otel_trace, "set_tracer_provider", installed.append)
    monkeypatch.setenv("OTEL_SERVICE_NAME", "adip-test")
    for name in tracing.ENDPOINT_ENVS:
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setattr(tracing, "_configured", False)
    tracing._configure_exporter()
    assert installed == []  # no endpoint: stay a no-op

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
    monkeypatch.setattr(tracing, "_configured", False)
    tracing._configure_exporter()
    [provider] = installed
    assert isinstance(provider, TracerProvider)
    assert provider.resource.attributes["service.name"] == "adip-test"
    provider.shutdown()

    installed.clear()
    monkeypatch.setattr(tracing, "_configured", False)
    monkeypatch.setattr(tracing.otel_trace, "get_tracer_provider", lambda: TracerProvider())
    tracing._configure_exporter()
    assert installed == []  # a provider the host already set up is left alone
