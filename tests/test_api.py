import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import adip.api.app as api_app
import adip.api.services as api_services
from adip.api.app import create_app
from adip.api.schemas import AgentRunRequest
from adip.api.services import (
    get_agent_trace,
    get_mlops_run,
    indexed_documents,
    list_agent_trace_history,
    list_mlops_run_history,
    retrieval_benchmark_summary,
)
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


def save_test_index(path: Path) -> None:
    chunks = [
        make_chunk("chunk_ingest", "The platform ingests documents and preserves source metadata."),
        make_chunk("chunk_agent", "The agent verifies evidence before writing cited answers."),
    ]
    build_index(chunks, backend="tfidf").save(path)


def test_api_health_endpoint():
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_dashboard_and_assets_are_served():
    client = TestClient(create_app())

    dashboard = client.get("/")
    script = client.get("/static/dashboard.js")
    styles = client.get("/static/dashboard.css")

    assert dashboard.status_code == 200
    assert "Agentic Document Intelligence" in dashboard.text
    assert "Upload Document" in dashboard.text
    assert "Quality" in dashboard.text
    assert "Export" in dashboard.text
    assert "Add Model" in dashboard.text
    assert "Model API Name" in dashboard.text
    assert "customModelStatus" in dashboard.text
    assert "Target Document" in dashboard.text
    assert "dashboard.js?v=" in dashboard.text
    assert "Test Model" in dashboard.text
    assert "API Key" in dashboard.text
    assert script.status_code == 200
    assert styles.status_code == 200


def test_api_model_profiles_endpoint_lists_profiles():
    client = TestClient(create_app())

    response = client.get("/model-profiles")
    payload = response.json()

    assert response.status_code == 200
    assert payload["count"] >= 3
    assert any(item["profile_id"] == "qwen3_8b_default" for item in payload["items"])
    cloud_profile = next(item for item in payload["items"] if item["profile_id"] == "deepseek_v4_pro_cloud")
    assert cloud_profile["runtime"]["api_key_env"] == "DEEPSEEK_API_KEY"
    assert "api_key" not in cloud_profile["runtime"]


def test_api_model_check_endpoint(monkeypatch):
    class FakeAdapter:
        def __init__(self, model_name, endpoint_url, api_key, timeout_seconds=30):
            self.model_name = model_name
            self.endpoint_url = endpoint_url
            self.api_key = api_key

        def generate(self, request_payload):
            return SimpleNamespace(
                text="OK",
                model_name=self.model_name,
                input_token_count=3,
                output_token_count=1,
                raw={"endpoint_url": self.endpoint_url},
            )

    monkeypatch.setattr(api_services, "OpenAICompatibleChatAdapter", FakeAdapter)
    client = TestClient(create_app())

    response = client.post(
        "/models/check",
        json={
            "model_name": "llama-3.1-8b-instant",
            "endpoint_url": "https://api.groq.com/openai/v1/chat/completions",
            "api_key": "secret",
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["model_name"] == "llama-3.1-8b-instant"
    assert payload["preview"] == "OK"


def test_api_rag_query_returns_cited_answer(tmp_path):
    index_path = tmp_path / "index"
    save_test_index(index_path)
    client = TestClient(create_app())

    response = client.post(
        "/rag/query",
        json={
            "index_path": str(index_path),
            "question": "What does the platform preserve?",
            "top_k": 1,
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["backend"] == "tfidf"
    assert payload["retrieved"][0]["chunk"]["chunk_id"] == "chunk_ingest"
    assert "sample.md" in payload["answer"]
    assert "quality" in payload
    assert payload["quality"]["evidence_count"] == 1


def test_api_rag_query_accepts_document_filter(tmp_path):
    index_path = tmp_path / "index"
    chunks = [
        {
            **make_chunk(
                "chunk_pdf",
                "The selected PDF discusses neural differential cryptanalysis.",
            ),
            "document_id": "doc_pdf",
            "filename": "paper.pdf",
            "source_type": "pdf",
        },
        {
            **make_chunk(
                "chunk_md",
                "The notes mention neural differential cryptanalysis.",
            ),
            "document_id": "doc_md",
            "filename": "notes.md",
        },
    ]
    build_index(chunks, backend="tfidf").save(index_path)
    client = TestClient(create_app())

    response = client.post(
        "/rag/query",
        json={
            "index_path": str(index_path),
            "document_filter": "doc_md",
            "question": "What mentions neural differential cryptanalysis?",
            "top_k": 1,
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["document_filter"] == "doc_md"
    assert payload["retrieved"][0]["chunk"]["filename"] == "notes.md"


def test_indexed_documents_service_lists_index_documents(tmp_path):
    index_path = tmp_path / "index"
    save_test_index(index_path)

    payload = indexed_documents(index_path)

    assert payload["document_count"] == 1
    assert payload["items"][0]["filename"] == "sample.md"
    assert payload["items"][0]["chunk_count"] == 2


def test_api_agent_run_writes_trace(tmp_path):
    index_path = tmp_path / "index"
    trace_dir = tmp_path / "traces"
    save_test_index(index_path)
    client = TestClient(create_app())

    response = client.post(
        "/agent/run",
        json={
            "index_path": str(index_path),
            "trace_dir": str(trace_dir),
            "question": "What does the agent verify?",
            "task": "qa",
            "top_k": 1,
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["state"]["status"] == "completed"
    assert payload["trace_path"]
    assert "quality" in payload
    assert payload["quality"]["evidence_count"] == 1
    assert Path(payload["trace_path"]).exists()


def test_api_upload_document_saves_supported_file(tmp_path):
    raw_dir = tmp_path / "raw"
    client = TestClient(create_app())

    response = client.post(
        "/documents/upload",
        data={"raw_dir": str(raw_dir)},
        files={"file": ("research brief.md", b"# Research\nUseful evidence.", "text/markdown")},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "uploaded"
    assert payload["filename"] == "research brief.md"
    assert payload["extension"] == ".md"
    assert Path(payload["path"]).read_text(encoding="utf-8") == "# Research\nUseful evidence."


def test_api_upload_document_rejects_unsupported_file(tmp_path):
    client = TestClient(create_app())

    response = client.post(
        "/documents/upload",
        data={"raw_dir": str(tmp_path / "raw")},
        files={"file": ("run.exe", b"nope", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "Unsupported document type" in response.json()["detail"]


def test_api_rebuild_index_runs_ingestion_and_indexing(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "sample.md").write_text(
        "The API rebuild endpoint parses documents, writes chunks, and saves a searchable index.",
        encoding="utf-8",
    )
    chunks_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "index"
    client = TestClient(create_app())

    response = client.post(
        "/pipeline/rebuild-index",
        json={
            "input_path": str(raw_dir),
            "chunks_path": str(chunks_path),
            "index_path": str(index_path),
            "chunk_size": 20,
            "chunk_overlap": 2,
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "completed"
    assert payload["ingestion"]["chunk_count"] >= 1
    assert payload["index"]["backend"] == "tfidf"
    assert chunks_path.exists()
    assert index_path.exists()


def test_api_list_documents_reports_index_status(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "indexed.md").write_text("Evidence text that will be indexed for retrieval.", encoding="utf-8")
    (raw_dir / "ignored.zip").write_bytes(b"not a document")
    chunks_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "index"
    client = TestClient(create_app())
    client.post(
        "/pipeline/rebuild-index",
        json={"input_path": str(raw_dir), "chunks_path": str(chunks_path), "index_path": str(index_path)},
    )
    (raw_dir / "pending.md").write_text("Uploaded after the rebuild, so not indexed yet.", encoding="utf-8")

    response = client.get("/documents", params={"raw_dir": str(raw_dir), "index_path": str(index_path)})

    payload = response.json()
    assert response.status_code == 200
    by_name = {item["filename"]: item for item in payload["items"]}
    assert set(by_name) == {"indexed.md", "pending.md"}  # .zip is not listed
    assert by_name["indexed.md"]["indexed"] is True
    assert by_name["pending.md"]["indexed"] is False
    assert payload["index_stale"] is True


def test_api_list_documents_works_without_an_index(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "only.md").write_text("No index has been built yet.", encoding="utf-8")
    client = TestClient(create_app())

    response = client.get(
        "/documents",
        params={"raw_dir": str(raw_dir), "index_path": str(tmp_path / "missing_index")},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["document_count"] == 1
    assert payload["items"][0]["indexed"] is False


def test_api_delete_document_removes_file_and_flags_stale_index(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "doomed.md").write_text("This document will be deleted.", encoding="utf-8")
    chunks_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "index"
    client = TestClient(create_app())
    client.post(
        "/pipeline/rebuild-index",
        json={"input_path": str(raw_dir), "chunks_path": str(chunks_path), "index_path": str(index_path)},
    )

    response = client.delete("/documents/doomed.md", params={"raw_dir": str(raw_dir)})

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "deleted"
    assert not (raw_dir / "doomed.md").exists()

    listing = client.get(
        "/documents", params={"raw_dir": str(raw_dir), "index_path": str(index_path)}
    ).json()
    assert listing["document_count"] == 0
    assert listing["indexed_but_deleted"] == ["doomed.md"]
    assert listing["index_stale"] is True

    repeat = client.delete("/documents/doomed.md", params={"raw_dir": str(raw_dir)})
    assert repeat.status_code == 404


def test_api_delete_document_rejects_unsupported_type(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "config.yaml").write_text("secret: true", encoding="utf-8")
    client = TestClient(create_app())

    response = client.delete("/documents/config.yaml", params={"raw_dir": str(raw_dir)})

    assert response.status_code == 400
    assert (raw_dir / "config.yaml").exists()


def test_delete_raw_document_cannot_escape_raw_dir(tmp_path):
    from adip.api.services import delete_raw_document

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    outside = tmp_path / "escape.md"
    outside.write_text("Lives outside the raw folder.", encoding="utf-8")

    try:
        delete_raw_document("../escape.md", raw_dir=raw_dir)
    except FileNotFoundError:
        pass  # the traversal component must be stripped, leaving a missing raw file
    assert outside.exists()


def test_retrieval_benchmark_summary_reads_selected_metrics(tmp_path):
    report_path = tmp_path / "report.json"
    metrics_path = tmp_path / "metrics.json"
    report_path.write_text(
        json.dumps(
            {
                "best_backend_by_mrr": "dense_lsa",
                "best_variant_by_mrr": "tfidf_cross_encoder_rerank",
                "candidate_k": 10,
                "chunk_count": 10,
                "rerankers": ["none", "lexical", "cross_encoder"],
                "top_k": 3,
            }
        ),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(
            {
                "tfidf_mrr": 0.733,
                "tfidf_cross_encoder_rerank_mrr": 0.9,
                "unused_metric": 123,
            }
        ),
        encoding="utf-8",
    )

    summary = retrieval_benchmark_summary(report_path=report_path, metrics_path=metrics_path)

    assert summary["available"] is True
    assert summary["best_variant_by_mrr"] == "tfidf_cross_encoder_rerank"
    assert summary["metrics"]["tfidf_cross_encoder_rerank_mrr"] == 0.9
    assert "unused_metric" not in summary["metrics"]


def test_history_services_summarize_agent_traces_and_mlops_runs(tmp_path):
    trace_dir = tmp_path / "agent_traces"
    trace_dir.mkdir()
    trace_path = trace_dir / "agent_test.json"
    trace_path.write_text(
        json.dumps(
            {
                "run_id": "agent_test",
                "status": "completed",
                "question": "What happened?",
                "task_type": "qa",
                "requested_task": "qa",
                "domain_preset": "general",
                "top_k": 2,
                "retrieved": [{"chunk": {"chunk_id": "c1"}}],
                "citations": ["sample.md p.1"],
                "final_answer": "Answer",
                "metrics": {"workflow_duration_ms": 12.5},
                "trace": [
                    {"node_name": "planner", "started_at": "t1", "ended_at": "t2"},
                    {"node_name": "writer", "started_at": "t3", "ended_at": "t4"},
                ],
            }
        ),
        encoding="utf-8",
    )

    run_dir = tmp_path / "mlops_runs"
    run_path = run_dir / "run_test" / "run.json"
    run_path.parent.mkdir(parents=True)
    run_path.write_text(
        json.dumps(
            {
                "run_id": "run_test",
                "run_name": "retrieval_backend_benchmark",
                "status": "completed",
                "started_at": "t1",
                "ended_at": "t2",
                "duration_ms": 25.0,
                "tags": {"pipeline": "rag"},
                "params": {"best_variant_by_mrr": "tfidf_cross_encoder_rerank"},
                "metrics": {"tfidf_cross_encoder_rerank_mrr": 0.9, "chunk_count": 10},
                "artifacts": {"metrics": "metrics.json"},
            }
        ),
        encoding="utf-8",
    )

    traces = list_agent_trace_history(trace_dir=trace_dir)
    trace_detail = get_agent_trace("agent_test", trace_dir=trace_dir)
    runs = list_mlops_run_history(run_dir=run_dir)
    run_detail = get_mlops_run("run_test", run_dir=run_dir)

    assert traces["items"][0]["run_id"] == "agent_test"
    assert traces["items"][0]["trace_event_count"] == 2
    assert trace_detail["trace"]["question"] == "What happened?"
    assert runs["items"][0]["key_metrics"]["tfidf_cross_encoder_rerank_mrr"] == 0.9
    assert run_detail["run"]["run_name"] == "retrieval_backend_benchmark"


def test_api_history_endpoints(monkeypatch):
    monkeypatch.setattr(
        api_app,
        "list_agent_trace_history",
        lambda limit=25: {"count": 1, "items": [{"run_id": "agent_test"}]},
    )
    monkeypatch.setattr(
        api_app,
        "get_agent_trace",
        lambda run_id: {"summary": {"run_id": run_id}, "trace": {"run_id": run_id}},
    )
    monkeypatch.setattr(
        api_app,
        "list_mlops_run_history",
        lambda limit=25: {"count": 1, "items": [{"run_id": "run_test"}]},
    )
    monkeypatch.setattr(
        api_app,
        "get_mlops_run",
        lambda run_id: {"summary": {"run_id": run_id}, "run": {"run_id": run_id}},
    )
    client = TestClient(create_app())

    assert client.get("/history/agent-traces").json()["items"][0]["run_id"] == "agent_test"
    assert client.get("/history/agent-traces/agent_test").json()["trace"]["run_id"] == "agent_test"
    assert client.get("/history/mlops-runs").json()["items"][0]["run_id"] == "run_test"
    assert client.get("/history/mlops-runs/run_test").json()["run"]["run_id"] == "run_test"


def test_agent_service_passes_device_and_max_tokens(monkeypatch):
    captured = {}

    def fake_run_agent_from_index_path(**kwargs):
        captured.update(kwargs)
        state = SimpleNamespace(to_dict=lambda: {"status": "completed", "final_answer": "ok"})
        return SimpleNamespace(to_dict=lambda: {"state": state.to_dict(), "trace_path": None})

    monkeypatch.setattr(api_services, "run_agent_from_index_path", fake_run_agent_from_index_path)

    payload = api_services.run_agent_workflow(
        AgentRunRequest(
            question="Summarize this.",
            index_path=Path("data/processed/vector_index"),
            document_filter="doc_selected",
            llm_provider="openai_compatible",
            model_profile=None,
            model_name="custom-writer-model",
            endpoint_url="https://api.example.test/v1",
            api_key="writer-secret",
            device="cuda:1",
            max_new_tokens=128,
            reasoning_effort="none",
            reasoning_provider="openai_compatible",
            reasoning_model_profile="extractive_baseline",
            reasoning_model_name="custom-reasoning-model",
            reasoning_endpoint_url="https://reasoning.example.test/v1",
            reasoning_api_key="reasoning-secret",
            reasoning_device="cuda:1",
            reasoning_max_new_tokens=64,
            use_reasoning_planner=True,
        )
    )

    assert payload["state"]["final_answer"] == "ok"
    assert captured["device"] == "cuda:1"
    assert captured["max_new_tokens"] == 128
    assert captured["reasoning_effort"] == "none"
    assert captured["document_filter"] == "doc_selected"
    assert captured["llm_provider"] == "openai_compatible"
    assert captured["model_profile"] is None
    assert captured["model_name"] == "custom-writer-model"
    assert captured["endpoint_url"] == "https://api.example.test/v1"
    assert captured["api_key"] == "writer-secret"
    assert captured["reasoning_provider"] == "openai_compatible"
    assert captured["reasoning_model_profile"] == "extractive_baseline"
    assert captured["reasoning_model_name"] == "custom-reasoning-model"
    assert captured["reasoning_endpoint_url"] == "https://reasoning.example.test/v1"
    assert captured["reasoning_api_key"] == "reasoning-secret"
    assert captured["reasoning_device"] == "cuda:1"
    assert captured["reasoning_max_new_tokens"] == 64
    assert captured["use_reasoning_planner"] is True


def test_agent_workflow_surfaces_answer_warning(monkeypatch):
    def fake_run_agent_from_index_path(**kwargs):
        state = {
            "status": "completed",
            "final_answer": "partial",
            "llmops": {"answer_warning": "This answer was cut off. Raise Max Tokens."},
        }
        return SimpleNamespace(to_dict=lambda: {"state": state, "trace_path": None})

    monkeypatch.setattr(api_services, "run_agent_from_index_path", fake_run_agent_from_index_path)

    payload = api_services.run_agent_workflow(AgentRunRequest(question="Summarize this."))

    assert payload["answer_warning"] == "This answer was cut off. Raise Max Tokens."


def test_agent_workflow_answer_warning_is_none_when_not_truncated(monkeypatch):
    def fake_run_agent_from_index_path(**kwargs):
        state = {"status": "completed", "final_answer": "done", "llmops": {"answer_warning": None}}
        return SimpleNamespace(to_dict=lambda: {"state": state, "trace_path": None})

    monkeypatch.setattr(api_services, "run_agent_from_index_path", fake_run_agent_from_index_path)

    payload = api_services.run_agent_workflow(AgentRunRequest(question="Summarize this."))

    assert payload["answer_warning"] is None


def test_generation_eval_summary_reads_metrics(tmp_path):
    metrics_path = tmp_path / "gen_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "gen_eval_case_count": 15.0,
                "gen_eval_mean_faithfulness": 0.68,
                "gen_eval_grounded_rate": 1.0,
                "gen_eval_mean_expected_coverage": 0.9,
                "gen_eval_mean_citation_coverage": 0.61,
                "gen_eval_refusal_rate": 0.0,
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "gen_report.json"
    report_path.write_text(
        json.dumps({"config": {"model_profile": "extractive_baseline", "task": "qa"}}),
        encoding="utf-8",
    )

    summary = api_services.generation_eval_summary(metrics_path=metrics_path, report_path=report_path)

    assert summary["available"] is True
    assert summary["faithfulness"] == 0.68
    assert summary["grounded_rate"] == 1.0
    assert summary["model_profile"] == "extractive_baseline"


def test_generation_eval_summary_absent_when_no_metrics(tmp_path):
    summary = api_services.generation_eval_summary(metrics_path=tmp_path / "missing.json")

    assert summary["available"] is False


def test_agent_request_defaults_to_second_gpu():
    request = AgentRunRequest(question="Summarize this.")

    assert request.device == "cuda:1"


def test_offline_eval_endpoint_serves_committed_snapshot():
    client = TestClient(create_app())

    response = client.get("/monitoring/offline-eval")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["judge"]["judge_model"]
    assert 0.0 <= payload["ragas"]["ragas_mean_faithfulness"] <= 1.0
    assert payload["ragas"]["answer_relevancy_caveat"]


def test_offline_eval_snapshot_gracefully_absent(tmp_path):
    summary = api_services.offline_eval_snapshot(snapshot_path=tmp_path / "missing.json")

    assert summary["available"] is False


def test_offline_eval_snapshot_gracefully_malformed(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")

    summary = api_services.offline_eval_snapshot(snapshot_path=broken)

    assert summary["available"] is False


def test_generation_eval_summary_includes_full_metrics(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        '{"gen_eval_mean_faithfulness": 0.6, "gen_eval_refusal_recall": 0.5}', encoding="utf-8"
    )

    summary = api_services.generation_eval_summary(metrics_path=metrics_path)

    assert summary["available"] is True
    assert summary["metrics"]["gen_eval_refusal_recall"] == 0.5


def test_dashboard_serves_new_ui_sections():
    client = TestClient(create_app())

    dashboard = client.get("/")

    body = dashboard.text
    assert "coldStartBanner" in body
    assert "ragBackend" in body
    assert "ragCompareBackends" in body
    assert 'data-mode="eval"' in body
    assert "compareView" in body


def test_api_rag_query_supports_keyword_rewriter(tmp_path):
    index_path = tmp_path / "index"
    save_test_index(index_path)
    client = TestClient(create_app())

    response = client.post(
        "/rag/query",
        json={
            "index_path": str(index_path),
            "question": "What do the platforms preserve?",
            "top_k": 1,
            "rewriter": "keywords",
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["rewriter"] == "keywords"
    assert payload["query_variants"][0] == "What do the platforms preserve?"
    assert len(payload["query_variants"]) >= 2
    assert payload["retrieved"][0]["chunk"]["chunk_id"] == "chunk_ingest"


def test_index_cache_reloads_when_index_file_changes(tmp_path):
    import time as time_module

    from adip.api.cache import IndexCache

    index_path = tmp_path / "index"
    save_test_index(index_path)
    cache = IndexCache(max_entries=2)

    first = cache.get(index_path)
    second = cache.get(index_path)
    assert first is second
    assert cache.stats()["hits"] == 1

    time_module.sleep(0.01)
    save_test_index(index_path)  # rebuild -> new mtime
    third = cache.get(index_path)
    assert third is not first
    assert cache.stats()["misses"] == 2


def test_query_cache_lru_eviction_and_stats():
    from adip.api.cache import QueryCache

    cache = QueryCache(max_entries=2)
    cache.put("a", {"answer": 1})
    cache.put("b", {"answer": 2})
    assert cache.get("a") == {"answer": 1}  # refresh 'a'
    cache.put("c", {"answer": 3})  # evicts 'b'

    assert cache.get("b") is None
    assert cache.get("c") == {"answer": 3}
    stats = cache.stats()
    assert stats["entries"] == 2
    assert stats["hits"] == 2 and stats["misses"] == 1


def test_api_rag_query_uses_cache_on_repeat_and_invalidates_on_rebuild(tmp_path):
    import time as time_module

    index_path = tmp_path / "index"
    save_test_index(index_path)
    client = TestClient(create_app())
    payload = {
        "index_path": str(index_path),
        "question": "What does the platform preserve?",
        "top_k": 1,
    }

    first = client.post("/rag/query", json=payload).json()
    second = client.post("/rag/query", json=payload).json()
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["answer"] == first["answer"]

    time_module.sleep(0.01)
    save_test_index(index_path)  # rebuild invalidates via mtime key
    third = client.post("/rag/query", json=payload).json()
    assert third["cached"] is False


def test_api_rag_query_cache_can_be_bypassed(tmp_path):
    index_path = tmp_path / "index"
    save_test_index(index_path)
    client = TestClient(create_app())
    payload = {
        "index_path": str(index_path),
        "question": "What does the platform preserve?",
        "top_k": 1,
        "use_cache": False,
    }

    first = client.post("/rag/query", json=payload).json()
    second = client.post("/rag/query", json=payload).json()
    assert first["cached"] is False
    assert second["cached"] is False


def test_cache_monitoring_endpoint_reports_stats():
    client = TestClient(create_app())

    response = client.get("/monitoring/cache")

    assert response.status_code == 200
    payload = response.json()
    assert "index_cache" in payload and "query_cache" in payload
    assert payload["query_cache"]["max_entries"] >= 1
    assert "Redis" in payload["scope"]


def test_drift_monitoring_endpoint_graceful_without_baseline(tmp_path, monkeypatch):
    import adip.monitoring.drift as drift_module

    monkeypatch.setattr(drift_module, "DEFAULT_BASELINE", tmp_path / "missing.json")
    client = TestClient(create_app())

    response = client.get("/monitoring/drift")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert "rebuild-baseline" in payload["reason"]


def test_rag_query_appends_to_query_log(tmp_path, monkeypatch):
    import adip.monitoring.drift as drift_module

    log_path = tmp_path / "query_log.jsonl"
    monkeypatch.setattr(drift_module, "DEFAULT_QUERY_LOG", log_path)
    index_path = tmp_path / "index"
    save_test_index(index_path)
    client = TestClient(create_app())

    client.post(
        "/rag/query",
        json={"index_path": str(index_path), "question": "What does the platform preserve?", "top_k": 1},
    )

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    import json as json_module

    record = json_module.loads(lines[0])
    assert record["question"] == "What does the platform preserve?"
    assert record["top_score"] > 0
    assert record["cached"] is False
