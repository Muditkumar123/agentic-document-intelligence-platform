# Tracing

Every agent run can be exported as OpenTelemetry spans: one span for the run, one per
node (including each retry of the retriever and the verifier), one per retrieval search
and one per model call. Any OTLP collector can display them; Arize Phoenix and Langfuse
both render them as a tree with timings and token usage.

Tracing is off by default and costs nothing when off. The JSON traces the dashboard
replays are written either way.

## Turn it on

```bash
pip install -e ".[tracing]"
```

Then set the standard OpenTelemetry variables before starting the API or the CLI:

| Variable | Example | Meaning |
| --- | --- | --- |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:6006` | Collector base URL; `/v1/traces` is appended |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | `http://localhost:6006/v1/traces` | Full traces URL, used as is (takes precedence) |
| `OTEL_EXPORTER_OTLP_HEADERS` | `Authorization=Basic%20<base64>` | Auth headers, URL-encoded |
| `OTEL_SERVICE_NAME` | `adip` | Service name on every span (default `adip`) |

With the extra installed and neither endpoint variable set, nothing is exported. If the
process already has a tracer provider (for example under `opentelemetry-instrument`),
the spans go to that provider and nothing else is configured.

### Arize Phoenix (local)

```bash
pip install arize-phoenix
phoenix serve                      # UI at http://localhost:6006
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:6006 python -m adip.api
```

### Langfuse

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic%20$(printf '%s' "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" | base64 | tr -d '\n')"
python -m adip.api
```

Use `https://us.cloud.langfuse.com/api/public/otel` for the US region, or your own host
for a self-hosted Langfuse.

## What a run looks like

```
invoke_agent adip
├── agent.node intent_router
├── agent.node planner
├── agent.node retriever
│   ├── retrieval.search            one per sub-question / follow-up query
│   └── retrieval.search
├── agent.node evidence_verifier
├── agent.node retriever            only when the verifier asked for a retry
│   └── retrieval.search
├── agent.node evidence_verifier
├── agent.node writer
│   └── chat <model>
└── agent.node citation_checker
```

With a reasoning model configured, `chat` spans also appear under `planner` (planning and
question splitting) and `evidence_verifier`.

| Span | Attributes |
| --- | --- |
| `invoke_agent adip` | `gen_ai.operation.name`, `gen_ai.agent.name`, `adip.run_id`, `adip.engine`, `adip.task`, `adip.top_k`, `adip.index.backend`, `adip.decomposer`, `adip.reranker`, `adip.model_profile`; at the end `adip.status`, `adip.task_type`, `adip.retrieval.round_count`, `adip.retrieval.call_count`, `adip.retrieval.retry_count`, `adip.citation_count` |
| `agent.node <name>` | `adip.node`, `adip.status`, `adip.retrieved_count`, `adip.retrieval.round_count` |
| `retrieval.search` | `adip.retrieval.backend`, `adip.retrieval.top_k`, `adip.retrieval.reranker`, `adip.retrieval.document_filtered`, `adip.retrieval.result_count`, `adip.retrieval.top_score` |
| `chat <model>` | `gen_ai.operation.name`, `gen_ai.system`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.request.max_tokens`, `gen_ai.response.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `adip.task_type`, `adip.prompt.version`, `adip.evidence_count`, `adip.latency_ms` |

A failed node or run has an error status; an exception raised inside a span is recorded
on it.

## What is never recorded

Question text, prompts, retrieved document text, model output, endpoint URLs and API
keys stay out of span attributes. The spans carry ids, counts, scores, model names and
token usage only, so a collector can be shared without leaking document content.
`tests/test_tracing.py` checks this.

The code is in [`src/adip/tracing.py`](../src/adip/tracing.py).
