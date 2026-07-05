"""Application service functions used by the HTTP API."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adip import __version__
from adip.agents.runner import run_agent_from_index_path
from adip.api.schemas import (
    AgentRunRequest,
    ModelCheckRequest,
    RagQueryRequest,
    RebuildIndexRequest,
)
from adip.config.env import load_project_env
from adip.config.model_profiles import load_model_profiles, profile_runtime_status
from adip.ingestion.pipeline import ingest_path
from adip.llmops.evaluation import evaluate_generation
from adip.llmops.models import GenerationRequest, OpenAICompatibleChatAdapter
from adip.llmops.pipeline import build_evidence
from adip.rag.answer import build_extractive_answer
from adip.rag.chunks import read_chunks_jsonl
from adip.rag.rerank import rerank_results, resolve_candidate_k
from adip.rag.retriever import build_index, load_index, summarize_index_documents
from adip.rag.rewrite import retrieve_with_rewrites, rewrite_question

SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".txt", ".md"}


def health_payload() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "agentic-document-intelligence-api",
        "version": __version__,
    }


def retrieval_benchmark_summary(
    report_path: Path = Path("data/monitoring/retrieval_benchmark_report.json"),
    metrics_path: Path = Path("data/monitoring/retrieval_benchmark_metrics.json"),
) -> dict[str, Any]:
    if not report_path.exists() or not metrics_path.exists():
        return {
            "available": False,
            "report_path": str(report_path),
            "metrics_path": str(metrics_path),
        }

    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    selected_metrics = {
        key: metrics[key]
        for key in (
            "tfidf_mrr",
            "tfidf_lexical_rerank_mrr",
            "tfidf_cross_encoder_rerank_mrr",
            "dense_lsa_mrr",
            "dense_lsa_lexical_rerank_mrr",
            "dense_lsa_cross_encoder_rerank_mrr",
            "variant_count",
        )
        if key in metrics
    }
    return {
        "available": True,
        "best_backend_by_mrr": report.get("best_backend_by_mrr"),
        "best_variant_by_mrr": report.get("best_variant_by_mrr"),
        "candidate_k": report.get("candidate_k"),
        "chunk_count": report.get("chunk_count"),
        "golden_path": report.get("golden_path"),
        "rerankers": report.get("rerankers", []),
        "top_k": report.get("top_k"),
        "metrics": selected_metrics,
        "report_path": str(report_path),
        "metrics_path": str(metrics_path),
    }


def generation_eval_summary(
    metrics_path: Path = Path("data/monitoring/generation_eval_metrics.json"),
    report_path: Path = Path("data/monitoring/generation_eval_report.json"),
) -> dict[str, Any]:
    if not metrics_path.exists():
        return {"available": False, "metrics_path": str(metrics_path)}

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config: dict[str, Any] = {}
    if report_path.exists():
        try:
            config = json.loads(report_path.read_text(encoding="utf-8")).get("config", {})
        except (ValueError, OSError):
            config = {}

    return {
        "available": True,
        "case_count": metrics.get("gen_eval_case_count"),
        "answered_count": metrics.get("gen_eval_answered_count"),
        "faithfulness": metrics.get("gen_eval_mean_faithfulness"),
        "grounded_rate": metrics.get("gen_eval_grounded_rate"),
        "answer_relevance": metrics.get("gen_eval_mean_answer_relevance"),
        "expected_coverage": metrics.get("gen_eval_mean_expected_coverage"),
        "citation_coverage": metrics.get("gen_eval_mean_citation_coverage"),
        "refusal_rate": metrics.get("gen_eval_refusal_rate"),
        "model_profile": config.get("model_profile"),
        "task": config.get("task"),
        "metrics": metrics,
        "metrics_path": str(metrics_path),
        "report_path": str(report_path),
    }


def offline_eval_snapshot(
    snapshot_path: Path = Path("data/reference/offline_eval_snapshot.json"),
) -> dict[str, Any]:
    """Committed results of offline evaluations (LLM judge, RAGAS) for the dashboard.

    These evaluations call live models, so they cannot run inside the deterministic
    container build; the latest offline run is committed as a dated snapshot instead.
    """
    if not snapshot_path.exists():
        return {"available": False, "snapshot_path": str(snapshot_path)}
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {"available": False, "snapshot_path": str(snapshot_path)}
    return {"available": True, "snapshot_path": str(snapshot_path), **snapshot}


def model_profiles_summary() -> dict[str, Any]:
    load_project_env()
    profiles = load_model_profiles()
    return {
        "items": [model_profile_summary(profile) for profile in profiles.values()],
        "count": len(profiles),
    }


def model_profile_summary(profile) -> dict[str, Any]:
    payload = profile.to_dict()
    payload["runtime"] = profile_runtime_status(profile)
    return payload


def save_uploaded_document(
    filename: str,
    content: bytes,
    raw_dir: Path = Path("data/raw"),
) -> dict[str, Any]:
    if not content:
        raise ValueError("Uploaded file is empty")
    safe_name = safe_document_filename(filename)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))
        raise ValueError(f"Unsupported document type `{suffix}`. Supported: {supported}")

    destination_dir = raw_dir.expanduser()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / safe_name
    destination.write_bytes(content)
    return {
        "status": "uploaded",
        "filename": safe_name,
        "path": str(destination),
        "raw_dir": str(destination_dir),
        "size_bytes": len(content),
        "extension": suffix,
    }


def list_raw_documents(
    raw_dir: Path = Path("data/raw"),
    index_path: Path = Path("data/processed/vector_index"),
) -> dict[str, Any]:
    """List uploadable documents in the raw folder and whether each is in the index."""
    root = raw_dir.expanduser()
    items: list[dict[str, Any]] = []
    if root.is_dir():
        for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_UPLOAD_EXTENSIONS:
                continue
            stat = path.stat()
            items.append(
                {
                    "filename": path.name,
                    "extension": path.suffix.lower(),
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                }
            )

    indexed_filenames: set[str] = set()
    try:
        index = load_index(index_path)
        indexed_filenames = {str(doc["filename"]) for doc in summarize_index_documents(index.chunks)}
    except Exception:  # the index may not exist yet; listing must still work
        indexed_filenames = set()

    for item in items:
        item["indexed"] = item["filename"] in indexed_filenames
    on_disk = {item["filename"] for item in items}
    indexed_but_deleted = sorted(name for name in indexed_filenames if name not in on_disk)
    return {
        "raw_dir": str(root),
        "document_count": len(items),
        "items": items,
        "indexed_but_deleted": indexed_but_deleted,
        "index_stale": bool(indexed_but_deleted) or any(not item["indexed"] for item in items),
    }


def delete_raw_document(filename: str, raw_dir: Path = Path("data/raw")) -> dict[str, Any]:
    """Delete one document from the raw folder. The index keeps serving its chunks
    until the next rebuild, so callers should surface a rebuild reminder."""
    safe_name = safe_document_filename(filename)
    root = raw_dir.expanduser().resolve()
    target = (root / safe_name).resolve()
    if target.parent != root:
        raise ValueError("Filename escapes the raw documents directory")
    if target.suffix.lower() not in SUPPORTED_UPLOAD_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))
        raise ValueError(f"Unsupported document type `{target.suffix.lower()}`. Supported: {supported}")
    if not target.is_file():
        raise FileNotFoundError(f"Document not found: {safe_name}")
    target.unlink()
    return {
        "status": "deleted",
        "filename": safe_name,
        "raw_dir": str(root),
        "note": "Rebuild the index to remove this document's chunks from retrieval.",
    }


def list_agent_trace_history(
    trace_dir: Path = Path("data/monitoring/agent_traces"),
    limit: int = 25,
) -> dict[str, Any]:
    paths = sorted_history_paths(trace_dir, "*.json", limit=limit)
    return {
        "items": [agent_trace_summary(path) for path in paths],
        "count": len(paths),
        "trace_dir": str(trace_dir),
    }


def get_agent_trace(
    run_id: str,
    trace_dir: Path = Path("data/monitoring/agent_traces"),
) -> dict[str, Any]:
    path = history_record_path(trace_dir, run_id, suffix=".json")
    if not path.exists():
        raise FileNotFoundError(f"Agent trace not found: {run_id}")
    payload = read_json(path)
    return {
        "path": str(path),
        "summary": agent_trace_summary(path, payload=payload),
        "trace": payload,
    }


def list_mlops_run_history(
    run_dir: Path = Path("data/monitoring/mlops_runs"),
    limit: int = 25,
) -> dict[str, Any]:
    paths = sorted_history_paths(run_dir, "*/run.json", limit=limit)
    return {
        "items": [mlops_run_summary(path) for path in paths],
        "count": len(paths),
        "run_dir": str(run_dir),
    }


def get_mlops_run(
    run_id: str,
    run_dir: Path = Path("data/monitoring/mlops_runs"),
) -> dict[str, Any]:
    path = history_record_path(run_dir, run_id, child="run.json")
    if not path.exists():
        raise FileNotFoundError(f"MLOps run not found: {run_id}")
    payload = read_json(path)
    return {
        "path": str(path),
        "summary": mlops_run_summary(path, payload=payload),
        "run": payload,
    }


def run_rag_query(request: RagQueryRequest) -> dict[str, Any]:
    started = time.perf_counter()
    index = load_index(request.index_path)
    candidate_k = resolve_candidate_k(request.top_k, request.candidate_k, request.reranker)

    variants = rewrite_question(request.question, rewriter=request.rewriter)
    candidates = retrieve_with_rewrites(
        index,
        variants,
        top_k=candidate_k,
        document_filter=request.document_filter,
    )
    retrieved = rerank_results(
        request.question,
        candidates,
        reranker=request.reranker,
        top_k=request.top_k,
        original_score_weight=request.rerank_weight,
        cross_encoder_model=request.cross_encoder_model,
        cross_encoder_device=request.cross_encoder_device,
        cross_encoder_batch_size=request.cross_encoder_batch_size,
        cross_encoder_local_files_only=not request.allow_reranker_download,
    )
    answer = build_extractive_answer(request.question, retrieved)
    retrieved_records = [item.to_dict() for item in retrieved]
    latency_ms = (time.perf_counter() - started) * 1000

    return {
        "question": request.question,
        "answer": answer,
        "backend": index.backend,
        "index_path": str(request.index_path),
        "document_filter": request.document_filter,
        "top_k": request.top_k,
        "candidate_k": candidate_k,
        "reranker": request.reranker,
        "rewriter": request.rewriter,
        "query_variants": variants,
        "cross_encoder_model": request.cross_encoder_model if request.reranker == "cross_encoder" else None,
        "latency_ms": latency_ms,
        "quality": quality_summary(answer, retrieved_records),
        "retrieved": retrieved_records,
    }


def run_agent_workflow(request: AgentRunRequest) -> dict[str, Any]:
    started = time.perf_counter()
    result = run_agent_from_index_path(
        question=request.question,
        index_path=request.index_path,
        task=request.task,
        domain_preset=request.domain,
        top_k=request.top_k,
        document_filter=request.document_filter,
        llm_provider=request.llm_provider,
        model_name=request.model_name,
        model_profile=request.model_profile,
        endpoint_url=request.endpoint_url,
        api_key=request.api_key,
        device=request.device,
        max_new_tokens=request.max_new_tokens,
        reasoning_effort=request.reasoning_effort,
        reasoning_provider=request.reasoning_provider,
        reasoning_model_name=request.reasoning_model_name,
        reasoning_model_profile=request.reasoning_model_profile,
        reasoning_endpoint_url=request.reasoning_endpoint_url,
        reasoning_api_key=request.reasoning_api_key,
        reasoning_device=request.reasoning_device,
        reasoning_max_new_tokens=request.reasoning_max_new_tokens,
        use_reasoning_planner=request.use_reasoning_planner,
        trace_dir=request.trace_dir,
        engine=request.engine,
    )
    payload = result.to_dict()
    payload["latency_ms"] = (time.perf_counter() - started) * 1000
    payload["quality"] = agent_quality_summary(payload.get("state", {}))
    payload["answer_warning"] = agent_answer_warning(payload.get("state", {}))
    return payload


def indexed_documents(index_path: Path = Path("data/processed/vector_index")) -> dict[str, Any]:
    index = load_index(index_path)
    documents = summarize_index_documents(index.chunks)
    return {
        "index_path": str(index_path),
        "backend": index.backend,
        "document_count": len(documents),
        "chunk_count": len(index.chunks),
        "items": documents,
    }


def check_model_connection(request: ModelCheckRequest) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        adapter = OpenAICompatibleChatAdapter(
            model_name=request.model_name,
            endpoint_url=request.endpoint_url,
            api_key=request.api_key,
            timeout_seconds=30,
        )
        response = adapter.generate(
            GenerationRequest(
                prompt="Reply with exactly: OK",
                question="Connection test",
                task_type="qa",
                domain_preset="general",
                evidence=[],
                max_new_tokens=request.max_new_tokens,
            )
        )
    except Exception as exc:
        return {
            "ok": False,
            "status": "failed",
            "model_name": request.model_name,
            "endpoint_url": request.endpoint_url,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "error": str(exc),
        }

    return {
        "ok": True,
        "status": "ok",
        "model_name": response.model_name,
        "endpoint_url": response.raw.get("endpoint_url") if response.raw else request.endpoint_url,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "input_token_count": response.input_token_count,
        "output_token_count": response.output_token_count,
        "preview": response.text[:300],
    }


def rebuild_index(request: RebuildIndexRequest) -> dict[str, Any]:
    started = time.perf_counter()
    ingestion = ingest_path(
        input_path=request.input_path,
        output_path=request.chunks_path,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )
    chunks = read_chunks_jsonl(request.chunks_path)
    index = build_index(
        chunks,
        backend=request.backend,
        ngram_max=request.ngram_max,
        embedding_model=request.embedding_model,
        dense_dimensions=request.dense_dimensions,
        use_faiss=request.use_faiss,
        rrf_k=request.rrf_k,
        hybrid_dense_weight=request.hybrid_dense_weight,
    )
    index.save(request.index_path)
    latency_ms = (time.perf_counter() - started) * 1000

    return {
        "status": "completed",
        "latency_ms": latency_ms,
        "ingestion": ingestion.to_dict(),
        "index": {
            "backend": index.backend,
            "chunk_count": len(index.chunks),
            "embedding_model": index.embedding_model,
            "index_path": str(request.index_path),
            "metadata": index.metadata,
            "vocabulary_size": index.vocabulary_size,
        },
    }


def sorted_history_paths(root: Path, pattern: str, limit: int) -> list[Path]:
    if limit <= 0:
        raise ValueError("limit must be greater than 0")
    if not root.exists():
        return []
    return sorted(
        root.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:limit]


def history_record_path(root: Path, record_id: str, suffix: str = "", child: str | None = None) -> Path:
    if not record_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in record_id):
        raise ValueError("History record id may only contain letters, numbers, underscores, and hyphens")
    if child:
        return root / record_id / child
    return root / f"{record_id}{suffix}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def modified_at_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def agent_trace_summary(path: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    trace = payload or read_json(path)
    events = trace.get("trace", [])
    metrics = trace.get("metrics", {})
    return {
        "run_id": trace.get("run_id") or path.stem,
        "path": str(path),
        "modified_at": modified_at_iso(path),
        "status": trace.get("status"),
        "question": trace.get("question"),
        "task_type": trace.get("task_type"),
        "requested_task": trace.get("requested_task"),
        "domain_preset": trace.get("domain_preset"),
        "top_k": trace.get("top_k"),
        "model_profile": trace.get("model_profile"),
        "trace_event_count": len(events),
        "retrieved_count": len(trace.get("retrieved", [])),
        "citation_count": len(trace.get("citations", [])),
        "answer_char_count": len(trace.get("final_answer", "")),
        "workflow_duration_ms": metrics.get("workflow_duration_ms"),
        "started_at": events[0].get("started_at") if events else None,
        "ended_at": events[-1].get("ended_at") if events else None,
    }


def mlops_run_summary(path: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    run = payload or read_json(path)
    metrics = run.get("metrics", {})
    params = run.get("params", {})
    return {
        "run_id": run.get("run_id") or path.parent.name,
        "run_name": run.get("run_name"),
        "path": str(path),
        "modified_at": modified_at_iso(path),
        "status": run.get("status"),
        "started_at": run.get("started_at"),
        "ended_at": run.get("ended_at"),
        "duration_ms": run.get("duration_ms"),
        "tags": run.get("tags", {}),
        "metric_count": len(metrics),
        "param_count": len(params),
        "artifact_count": len(run.get("artifacts", {})),
        "key_params": selected_history_values(
            params,
            [
                "best_backend_by_mrr",
                "best_variant_by_mrr",
                "backend",
                "model_profile",
                "rerankers",
                "top_k",
            ],
        ),
        "key_metrics": selected_history_values(
            metrics,
            [
                "chunk_count",
                "question_count",
                "hit_rate_at_k",
                "mrr",
                "tfidf_mrr",
                "tfidf_cross_encoder_rerank_mrr",
                "dense_lsa_mrr",
                "llm_latency_ms",
                "workflow_duration_ms",
            ],
        ),
    }


def selected_history_values(values: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: values[key] for key in keys if key in values}


def safe_document_filename(filename: str) -> str:
    name = Path(filename or "").name.strip()
    if not name:
        raise ValueError("Uploaded file must have a filename")
    sanitized = re.sub(r"[^A-Za-z0-9._() -]+", "_", name)
    sanitized = sanitized.strip(" .")
    if not sanitized or sanitized in {".", ".."}:
        raise ValueError("Uploaded filename is not usable")
    return sanitized


def quality_summary(answer: str, retrieved_records: list[dict[str, Any]]) -> dict[str, Any]:
    evidence = build_evidence(retrieved_records)
    report = evaluate_generation(answer, evidence)
    unsupported_rate = (
        report.unsupported_sentence_count / report.answer_sentence_count
        if report.answer_sentence_count
        else 0.0
    )
    fidelity_score = max(0.0, report.citation_coverage - unsupported_rate)
    payload = report.to_dict()
    payload["fidelity_score"] = fidelity_score
    return payload


def agent_answer_warning(state_payload: dict[str, Any]) -> str | None:
    """Surface the LLMOps truncation/thinking-budget note for the dashboard."""
    llmops = state_payload.get("llmops") or {}
    return llmops.get("answer_warning")


def agent_quality_summary(state_payload: dict[str, Any]) -> dict[str, Any]:
    quality = state_payload.get("llmops", {}).get("quality")
    if quality:
        answer_count = float(quality.get("answer_sentence_count") or 0)
        unsupported_rate = (
            float(quality.get("unsupported_sentence_count") or 0) / answer_count
            if answer_count
            else 0.0
        )
        quality = dict(quality)
        quality["fidelity_score"] = max(0.0, float(quality.get("citation_coverage") or 0.0) - unsupported_rate)
        return quality
    return quality_summary(state_payload.get("final_answer", ""), state_payload.get("retrieved", []))
