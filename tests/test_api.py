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
from adip.api.settings import ApiSettings
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


ADMIN_TOKEN = "test-admin-token"
AUTH = {"Authorization": f"Bearer {ADMIN_TOKEN}"}


def make_settings(tmp_path: Path, **overrides) -> ApiSettings:
    """Settings rooted in a temp data dir, so tests never touch the repo's data/."""
    return ApiSettings(data_dir=tmp_path / "data", **{"admin_token": ADMIN_TOKEN, **overrides})


def make_client(settings: ApiSettings | None = None, **client_kwargs) -> TestClient:
    return TestClient(create_app(settings or ApiSettings(admin_token=ADMIN_TOKEN)), **client_kwargs)


def test_api_health_endpoint():
    client = make_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_dashboard_and_assets_are_served():
    client = make_client()

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
    client = make_client()

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
    client = make_client()

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
    settings = make_settings(tmp_path)
    save_test_index(settings.index_path())
    client = make_client(settings)

    response = client.post(
        "/rag/query",
        json={
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
    settings = make_settings(tmp_path)
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
    build_index(chunks, backend="tfidf").save(settings.index_path())
    client = make_client(settings)

    response = client.post(
        "/rag/query",
        json={
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
    settings = make_settings(tmp_path)
    save_test_index(settings.index_path())
    client = make_client(settings)

    response = client.post(
        "/agent/run",
        json={
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
    assert Path(payload["trace_path"]).parent == settings.trace_dir


def test_api_upload_document_saves_supported_file(tmp_path):
    settings = make_settings(tmp_path)
    client = make_client(settings)

    response = client.post(
        "/documents/upload",
        headers=AUTH,
        files={"file": ("research brief.md", b"# Research\nUseful evidence.", "text/markdown")},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "uploaded"
    assert payload["filename"] == "research brief.md"
    assert payload["extension"] == ".md"
    assert (settings.raw_dir / "research brief.md").read_text(encoding="utf-8") == "# Research\nUseful evidence."


def test_api_upload_document_rejects_unsupported_file(tmp_path):
    client = make_client(make_settings(tmp_path))

    response = client.post(
        "/documents/upload",
        headers=AUTH,
        files={"file": ("run.exe", b"nope", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "Unsupported document type" in response.json()["detail"]


def test_api_rebuild_index_runs_ingestion_and_indexing(tmp_path):
    settings = make_settings(tmp_path)
    settings.raw_dir.mkdir(parents=True)
    (settings.raw_dir / "sample.md").write_text(
        "The API rebuild endpoint parses documents, writes chunks, and saves a searchable index.",
        encoding="utf-8",
    )
    client = make_client(settings)

    response = client.post(
        "/pipeline/rebuild-index",
        headers=AUTH,
        json={"chunk_size": 20, "chunk_overlap": 2},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "completed"
    assert payload["ingestion"]["chunk_count"] >= 1
    assert payload["index"]["backend"] == "tfidf"
    assert settings.chunks_path.exists()
    assert settings.index_path("tfidf").exists()


def test_api_rebuild_index_all_backends_builds_siblings(tmp_path):
    settings = make_settings(tmp_path)
    settings.raw_dir.mkdir(parents=True)
    (settings.raw_dir / "sample.md").write_text(
        "Rebuilding with all_backends keeps the tfidf, dense, and hybrid indexes in sync.",
        encoding="utf-8",
    )
    client = make_client(settings)

    response = client.post(
        "/pipeline/rebuild-index",
        headers=AUTH,
        json={"use_faiss": False, "all_backends": True},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["index"]["backend"] == "tfidf"
    extra = {item["backend"]: item["index_path"] for item in payload["additional_indexes"]}
    assert set(extra) == {"dense_lsa", "hybrid"}
    assert settings.index_path("dense").exists()
    assert settings.index_path("hybrid").exists()


def test_api_rag_query_reports_unmatched_document_filter(tmp_path):
    settings = make_settings(tmp_path)
    chunks = [make_chunk("chunk_a", "Neural differential cryptanalysis notes.")]
    build_index(chunks, backend="tfidf").save(settings.index_path())
    client = make_client(settings)

    response = client.post(
        "/rag/query",
        json={
            "document_filter": "zzz-absent-doc",
            "question": "What mentions neural differential cryptanalysis?",
            "top_k": 1,
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["retrieved"] == []
    assert "does not match any document" in payload["answer"]
    assert "sample.md" in payload["answer"]


def test_api_list_documents_reports_index_status(tmp_path):
    settings = make_settings(tmp_path)
    raw_dir = settings.raw_dir
    raw_dir.mkdir(parents=True)
    (raw_dir / "indexed.md").write_text("Evidence text that will be indexed for retrieval.", encoding="utf-8")
    (raw_dir / "ignored.zip").write_bytes(b"not a document")
    client = make_client(settings)
    client.post("/pipeline/rebuild-index", headers=AUTH, json={})
    (raw_dir / "pending.md").write_text("Uploaded after the rebuild, so not indexed yet.", encoding="utf-8")

    response = client.get("/documents")

    payload = response.json()
    assert response.status_code == 200
    by_name = {item["filename"]: item for item in payload["items"]}
    assert set(by_name) == {"indexed.md", "pending.md"}  # .zip is not listed
    assert by_name["indexed.md"]["indexed"] is True
    assert by_name["pending.md"]["indexed"] is False
    assert payload["index_stale"] is True


def test_api_list_documents_works_without_an_index(tmp_path):
    settings = make_settings(tmp_path)
    settings.raw_dir.mkdir(parents=True)
    (settings.raw_dir / "only.md").write_text("No index has been built yet.", encoding="utf-8")
    client = make_client(settings)

    response = client.get("/documents")

    payload = response.json()
    assert response.status_code == 200
    assert payload["document_count"] == 1
    assert payload["items"][0]["indexed"] is False


def test_api_delete_document_removes_file_and_flags_stale_index(tmp_path):
    settings = make_settings(tmp_path)
    raw_dir = settings.raw_dir
    raw_dir.mkdir(parents=True)
    (raw_dir / "doomed.md").write_text("This document will be deleted.", encoding="utf-8")
    client = make_client(settings)
    client.post("/pipeline/rebuild-index", headers=AUTH, json={})

    response = client.delete("/documents/doomed.md", headers=AUTH)

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "deleted"
    assert not (raw_dir / "doomed.md").exists()

    listing = client.get("/documents").json()
    assert listing["document_count"] == 0
    assert listing["indexed_but_deleted"] == ["doomed.md"]
    assert listing["index_stale"] is True

    repeat = client.delete("/documents/doomed.md", headers=AUTH)
    assert repeat.status_code == 404


def test_api_delete_document_rejects_unsupported_type(tmp_path):
    settings = make_settings(tmp_path)
    settings.raw_dir.mkdir(parents=True)
    (settings.raw_dir / "config.yaml").write_text("secret: true", encoding="utf-8")
    client = make_client(settings)

    response = client.delete("/documents/config.yaml", headers=AUTH)

    assert response.status_code == 400
    assert (settings.raw_dir / "config.yaml").exists()


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
        lambda limit=25, **_dirs: {"count": 1, "items": [{"run_id": "agent_test"}]},
    )
    monkeypatch.setattr(
        api_app,
        "get_agent_trace",
        lambda run_id, **_dirs: {"summary": {"run_id": run_id}, "trace": {"run_id": run_id}},
    )
    monkeypatch.setattr(
        api_app,
        "list_mlops_run_history",
        lambda limit=25, **_dirs: {"count": 1, "items": [{"run_id": "run_test"}]},
    )
    monkeypatch.setattr(
        api_app,
        "get_mlops_run",
        lambda run_id, **_dirs: {"summary": {"run_id": run_id}, "run": {"run_id": run_id}},
    )
    client = make_client()

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
        ),
        index_path=Path("data/processed/vector_index"),
    )

    assert payload["state"]["final_answer"] == "ok"
    assert captured["index_path"] == Path("data/processed/vector_index")
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

    payload = api_services.run_agent_workflow(AgentRunRequest(question="Summarize this."), index_path=Path("idx"))

    assert payload["answer_warning"] == "This answer was cut off. Raise Max Tokens."


def test_agent_workflow_answer_warning_is_none_when_not_truncated(monkeypatch):
    def fake_run_agent_from_index_path(**kwargs):
        state = {"status": "completed", "final_answer": "done", "llmops": {"answer_warning": None}}
        return SimpleNamespace(to_dict=lambda: {"state": state, "trace_path": None})

    monkeypatch.setattr(api_services, "run_agent_from_index_path", fake_run_agent_from_index_path)

    payload = api_services.run_agent_workflow(AgentRunRequest(question="Summarize this."), index_path=Path("idx"))

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
    client = make_client()

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
    client = make_client()

    dashboard = client.get("/")

    body = dashboard.text
    assert "coldStartBanner" in body
    assert "ragBackend" in body
    assert "ragCompareBackends" in body
    assert 'data-mode="eval"' in body
    assert "compareView" in body


def test_api_rag_query_supports_keyword_rewriter(tmp_path):
    settings = make_settings(tmp_path)
    save_test_index(settings.index_path())
    client = make_client(settings)

    response = client.post(
        "/rag/query",
        json={
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

    settings = make_settings(tmp_path)
    index_path = settings.index_path()
    save_test_index(index_path)
    client = make_client(settings)
    payload = {
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
    settings = make_settings(tmp_path)
    save_test_index(settings.index_path())
    client = make_client(settings)
    payload = {
        "question": "What does the platform preserve?",
        "top_k": 1,
        "use_cache": False,
    }

    first = client.post("/rag/query", json=payload).json()
    second = client.post("/rag/query", json=payload).json()
    assert first["cached"] is False
    assert second["cached"] is False


def test_cache_monitoring_endpoint_reports_stats():
    client = make_client()

    response = client.get("/monitoring/cache")

    assert response.status_code == 200
    payload = response.json()
    assert "index_cache" in payload and "query_cache" in payload
    assert payload["query_cache"]["max_entries"] >= 1
    assert "Redis" in payload["scope"]


def test_drift_monitoring_endpoint_graceful_without_baseline(tmp_path, monkeypatch):
    import adip.monitoring.drift as drift_module

    monkeypatch.setattr(drift_module, "DEFAULT_BASELINE", tmp_path / "missing.json")
    client = make_client()

    response = client.get("/monitoring/drift")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert "rebuild-baseline" in payload["reason"]


def test_rag_query_appends_to_query_log(tmp_path, monkeypatch):
    import adip.monitoring.drift as drift_module

    log_path = tmp_path / "query_log.jsonl"
    monkeypatch.setattr(drift_module, "DEFAULT_QUERY_LOG", log_path)
    settings = make_settings(tmp_path)
    save_test_index(settings.index_path())
    client = make_client(settings)

    client.post("/rag/query", json={"question": "What does the platform preserve?", "top_k": 1})

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    import json as json_module

    record = json_module.loads(lines[0])
    assert record["question"] == "What does the platform preserve?"
    assert record["top_score"] > 0
    assert record["cached"] is False


# --- Security: server-owned paths, write protection, upload limits -----------------


def test_api_requests_reject_client_supplied_paths(tmp_path):
    settings = make_settings(tmp_path)
    save_test_index(settings.index_path())
    client = make_client(settings)

    query = client.post("/rag/query", json={"question": "q", "index_path": "/etc"})
    agent = client.post("/agent/run", json={"question": "q", "trace_dir": str(tmp_path / "elsewhere")})
    rebuild = client.post("/pipeline/rebuild-index", headers=AUTH, json={"input_path": "/etc"})

    assert query.status_code == 422
    assert agent.status_code == 422
    assert rebuild.status_code == 422
    assert not (tmp_path / "elsewhere").exists()


def test_api_upload_ignores_client_raw_dir(tmp_path):
    settings = make_settings(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    client = make_client(settings)

    response = client.post(
        "/documents/upload",
        headers=AUTH,
        data={"raw_dir": str(elsewhere)},
        files={"file": ("note.md", b"# Note", "text/markdown")},
    )

    assert response.status_code == 200
    assert (settings.raw_dir / "note.md").exists()
    assert not elsewhere.exists()


def test_api_delete_ignores_client_raw_dir(tmp_path):
    settings = make_settings(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "victim.md").write_text("Lives outside the server's raw folder.", encoding="utf-8")
    client = make_client(settings)

    response = client.delete("/documents/victim.md", headers=AUTH, params={"raw_dir": str(elsewhere)})

    assert response.status_code == 404
    assert (elsewhere / "victim.md").exists()


def test_api_rag_query_selects_index_by_backend_name(tmp_path):
    settings = make_settings(tmp_path)
    chunks = [
        make_chunk("chunk_ingest", "The platform ingests documents and preserves source metadata."),
        make_chunk("chunk_agent", "The agent verifies evidence before writing cited answers."),
    ]
    build_index(chunks, backend="tfidf").save(settings.index_path("tfidf"))
    build_index(chunks, backend="hybrid").save(settings.index_path("hybrid"))
    client = make_client(settings)

    tfidf = client.post("/rag/query", json={"question": "What does the agent verify?", "top_k": 1})
    hybrid = client.post(
        "/rag/query", json={"question": "What does the agent verify?", "top_k": 1, "backend": "hybrid"}
    )
    missing = client.post(
        "/rag/query", json={"question": "What does the agent verify?", "top_k": 1, "backend": "dense"}
    )

    assert tfidf.json()["backend"] == "tfidf"
    assert hybrid.json()["backend"] == "hybrid"
    assert missing.status_code == 404


def test_write_endpoints_require_the_admin_token(tmp_path):
    settings = make_settings(tmp_path)
    settings.raw_dir.mkdir(parents=True)
    (settings.raw_dir / "keep.md").write_text("Must survive unauthenticated deletes.", encoding="utf-8")
    client = make_client(settings)
    upload = {"files": {"file": ("x.md", b"# x", "text/markdown")}}
    wrong = {"Authorization": "Bearer not-the-token"}

    no_token = client.post("/documents/upload", **upload)
    bad_token = client.post("/documents/upload", headers=wrong, **upload)
    delete = client.delete("/documents/keep.md")
    rebuild = client.post("/pipeline/rebuild-index", json={})

    assert no_token.status_code == 401
    assert no_token.headers["www-authenticate"] == "Bearer"
    assert "admin token" in no_token.json()["detail"]
    assert bad_token.status_code == 403
    assert delete.status_code == 401
    assert rebuild.status_code == 401
    assert not (settings.raw_dir / "x.md").exists()
    assert (settings.raw_dir / "keep.md").exists()
    assert not settings.index_path().exists()


def test_write_endpoints_without_a_token_only_accept_loopback_clients(tmp_path):
    settings = make_settings(tmp_path, admin_token=None)
    upload = {"files": {"file": ("x.md", b"# x", "text/markdown")}}

    remote = make_client(settings, client=("203.0.113.7", 50000)).post("/documents/upload", **upload)
    local = make_client(settings, client=("127.0.0.1", 50000)).post("/documents/upload", **upload)

    assert remote.status_code == 403
    assert "ADIP_ADMIN_TOKEN" in remote.json()["detail"]
    assert local.status_code == 200


def test_read_endpoints_stay_open_when_a_token_is_configured(tmp_path):
    settings = make_settings(tmp_path)
    save_test_index(settings.index_path())
    client = make_client(settings, client=("203.0.113.7", 50000))

    assert client.get("/documents").status_code == 200
    assert client.get("/index/documents").status_code == 200
    assert client.post("/rag/query", json={"question": "What does the platform preserve?"}).status_code == 200


def test_health_reports_write_access_mode(tmp_path):
    with_token = make_client(make_settings(tmp_path)).get("/health").json()
    without = make_client(make_settings(tmp_path, admin_token=None)).get("/health").json()

    assert with_token["write_access"] == "admin_token"
    assert without["write_access"] == "local_only"
    assert with_token["max_upload_mb"] == 20


def test_upload_over_the_size_limit_is_rejected(tmp_path):
    settings = make_settings(tmp_path, max_upload_bytes=1024)
    client = make_client(settings)

    response = client.post(
        "/documents/upload",
        headers=AUTH,
        files={"file": ("big.md", b"x" * 2048, "text/markdown")},
    )

    assert response.status_code == 413
    assert not (settings.raw_dir / "big.md").exists()


def test_request_body_over_the_limit_is_rejected_before_auth(tmp_path):
    settings = make_settings(tmp_path, max_upload_bytes=1024)
    client = make_client(settings)
    oversized = b"x" * (settings.max_request_bytes + 1)

    response = client.post("/documents/upload", files={"file": ("big.md", oversized, "text/markdown")})

    # The size middleware answers before auth runs; without it this would be a 401.
    assert response.status_code == 413
    assert response.json()["detail"].startswith("Request is too large")


def test_streamed_request_body_without_content_length_is_counted(tmp_path):
    settings = make_settings(tmp_path, max_upload_bytes=1024)
    save_test_index(settings.index_path())
    client = make_client(settings)
    question = "x" * (settings.max_request_bytes + 1)

    def body():
        yield b'{"question": "'
        yield question.encode()
        yield b'"}'

    response = client.post("/rag/query", content=body(), headers={"Content-Type": "application/json"})

    assert response.status_code == 413
    assert response.json()["detail"].startswith("Request is too large")


def test_api_settings_from_env():
    settings = ApiSettings.from_env({"ADIP_ADMIN_TOKEN": "  secret  ", "ADIP_MAX_UPLOAD_MB": "5"})
    defaults = ApiSettings.from_env({"ADIP_ADMIN_TOKEN": "   "})

    assert settings.admin_token == "secret"
    assert settings.max_upload_bytes == 5 * 1024 * 1024
    assert defaults.admin_token is None
    assert defaults.max_upload_bytes == 20 * 1024 * 1024
    try:
        ApiSettings.from_env({"ADIP_MAX_UPLOAD_MB": "lots"})
    except ValueError as exc:
        assert "ADIP_MAX_UPLOAD_MB" in str(exc)
    else:
        raise AssertionError("an invalid upload limit must be rejected")


def test_agent_service_forwards_retrieval_loop_options(monkeypatch):
    captured = {}

    def fake_run_agent_from_index_path(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(to_dict=lambda: {"state": {"status": "completed", "final_answer": "ok"}, "trace_path": None})

    monkeypatch.setattr(api_services, "run_agent_from_index_path", fake_run_agent_from_index_path)

    api_services.run_agent_workflow(
        AgentRunRequest(question="Compare A and B.", decomposer="none", max_retrieval_retries=0),
        index_path=Path("idx"),
    )
    defaults = AgentRunRequest(question="q")

    assert captured["decomposer"] == "none"
    assert captured["max_retrieval_retries"] == 0
    assert defaults.decomposer == "rules"
    assert defaults.max_retrieval_retries == 2


def test_agent_run_rejects_out_of_range_retry_budget(tmp_path):
    client = make_client(make_settings(tmp_path))

    response = client.post("/agent/run", json={"question": "q", "max_retrieval_retries": 10})

    assert response.status_code == 422
