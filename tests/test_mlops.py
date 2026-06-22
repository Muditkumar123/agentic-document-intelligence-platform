import json

from adip.mlops.run_ingestion import main as run_ingestion_main
from adip.mlops.run_retrieval_benchmark import best_report_by_mrr
from adip.mlops.run_retrieval_benchmark import main as run_retrieval_benchmark_main
from adip.mlops.tracking import start_run


def test_start_run_writes_local_run_record(tmp_path):
    with start_run("unit_test_run", run_dir=tmp_path, tags={"pipeline": "test"}) as run:
        run.log_param("chunk_size", 128)
        run.log_metric("chunk_count", 3)

    run_record = json.loads(run.run_path.read_text(encoding="utf-8"))

    assert run_record["run_name"] == "unit_test_run"
    assert run_record["status"] == "completed"
    assert run_record["tags"]["pipeline"] == "test"
    assert run_record["params"]["chunk_size"] == 128
    assert run_record["metrics"]["chunk_count"] == 3.0
    assert run_record["mlflow_available"] is False


def test_tracked_ingestion_command_writes_metrics_and_run(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "sample.md").write_text(
        "This document should be chunked and tracked by the MLOps wrapper.",
        encoding="utf-8",
    )
    output_path = tmp_path / "processed" / "chunks.jsonl"
    metrics_path = tmp_path / "monitoring" / "ingestion_metrics.json"
    run_dir = tmp_path / "runs"

    exit_code = run_ingestion_main(
        [
            "--input",
            str(raw_dir),
            "--output",
            str(output_path),
            "--chunk-size",
            "8",
            "--chunk-overlap",
            "2",
            "--metrics-output",
            str(metrics_path),
            "--run-dir",
            str(run_dir),
        ]
    )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_records = list(run_dir.glob("*/run.json"))

    assert exit_code == 0
    assert output_path.exists()
    assert metrics["document_count"] == 1
    assert metrics["chunk_count"] >= 1
    assert len(run_records) == 1


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


def test_retrieval_benchmark_command_compares_backends(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks = [
        make_chunk("chunk_ingest", "The platform ingests documents and preserves source metadata."),
        make_chunk("chunk_agent", "The agent verifies evidence before writing answers."),
    ]
    chunks_path.write_text(
        "\n".join(json.dumps(chunk) for chunk in chunks) + "\n",
        encoding="utf-8",
    )
    golden_path = tmp_path / "golden.jsonl"
    golden_path.write_text(
        json.dumps(
            {
                "question": "What does the platform preserve?",
                "expected_chunk_ids": ["chunk_ingest"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metrics_path = tmp_path / "monitoring" / "retrieval_benchmark_metrics.json"
    report_path = tmp_path / "monitoring" / "retrieval_benchmark_report.json"
    run_dir = tmp_path / "runs"

    exit_code = run_retrieval_benchmark_main(
        [
            "--chunks",
            str(chunks_path),
            "--index-root",
            str(tmp_path / "indexes"),
            "--golden",
            str(golden_path),
            "--backends",
            "tfidf",
            "dense",
            "--rerankers",
            "none",
            "lexical",
            "--embedding-model",
            "lsa",
            "--dense-dimensions",
            "8",
            "--no-faiss",
            "--top-k",
            "2",
            "--candidate-k",
            "2",
            "--metrics-output",
            str(metrics_path),
            "--report-output",
            str(report_path),
            "--run-dir",
            str(run_dir),
        ]
    )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    run_records = list(run_dir.glob("*/run.json"))

    assert exit_code == 0
    assert "tfidf_mrr" in metrics
    assert "dense_lsa_mrr" in metrics
    assert "dense_lsa_minus_tfidf_mrr" in metrics
    assert "tfidf_lexical_rerank_mrr" in metrics
    assert "dense_lsa_lexical_rerank_minus_plain_mrr" in metrics
    assert report["backend_count"] == 2
    assert report["variant_count"] == 4
    assert set(report["backends"]) == {"tfidf", "dense_lsa"}
    assert set(report["variants"]) == {
        "tfidf",
        "tfidf_lexical_rerank",
        "dense_lsa",
        "dense_lsa_lexical_rerank",
    }
    assert len(run_records) == 1


def test_best_report_by_mrr_uses_hit_rate_as_tiebreaker():
    reports = {
        "tfidf": {"mrr": 0.7, "hit_rate_at_k": 1.0},
        "dense_lsa": {"mrr": 0.8, "hit_rate_at_k": 0.8},
        "cross_encoder": {"mrr": 0.8, "hit_rate_at_k": 1.0},
    }

    assert best_report_by_mrr(reports) == "cross_encoder"
