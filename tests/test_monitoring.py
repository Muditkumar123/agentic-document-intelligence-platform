import json

import pytest

from adip.monitoring.drift import (
    QueryRecord,
    append_query_record,
    build_baseline,
    drift_report,
    population_stability_index,
    read_query_log,
)


def make_records(questions_scores):
    return [QueryRecord(question=q, top_score=s) for q, s in questions_scores]


BASELINE_RECORDS = make_records(
    [
        ("What are the principles for processing personal data?", 0.62),
        ("Which HTTP request methods are considered safe?", 0.58),
        ("What does the TLS protocol protect against?", 0.66),
        ("How does diversification reduce investment risk?", 0.51),
        ("What are the core functions of the cybersecurity framework?", 0.55),
        ("What architecture did the attention paper propose?", 0.60),
        ("What is the maximum length of a domain name?", 0.57),
        ("Which category of AI systems is prohibited outright?", 0.63),
        ("What does a mutual fund share represent?", 0.54),
        ("How does residual learning ease training?", 0.59),
    ]
)


def test_build_baseline_summarizes_reference_distribution():
    baseline = build_baseline(BASELINE_RECORDS)

    assert baseline["query_count"] == 10
    assert baseline["length_mean"] > 0
    assert len(baseline["score_bin_edges"]) == 9
    assert abs(sum(baseline["score_bin_proportions"]) - 1.0) < 1e-9
    assert "personal" in baseline["vocabulary"]


def test_build_baseline_rejects_too_few_records():
    with pytest.raises(ValueError):
        build_baseline(BASELINE_RECORDS[:1])


def test_psi_is_near_zero_for_identical_distributions():
    proportions = [0.1] * 10
    assert population_stability_index(proportions, proportions) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_for_shifted_distributions():
    expected = [0.1] * 10
    shifted = [0.0] * 9 + [1.0]
    assert population_stability_index(expected, shifted) > 0.25


def test_drift_report_ok_when_queries_match_baseline():
    baseline = build_baseline(BASELINE_RECORDS)
    report = drift_report(baseline, BASELINE_RECORDS[:6])

    assert report["available"] is True
    assert report["overall_status"] == "ok"
    assert report["components"]["vocabulary_oov_rate"]["value"] < 0.35


def test_drift_report_alerts_on_off_distribution_queries():
    baseline = build_baseline(BASELINE_RECORDS)
    weird = make_records(
        [
            ("zebra quantum spaghetti volcano xylophone", 0.02),
            ("banana helicopter jazz tornado bicycle", 0.01),
            ("penguin lasagna asteroid trombone circus", 0.03),
            ("waffle dinosaur telescope harmonica glacier", 0.02),
            ("cactus submarine violin meteor pretzel", 0.01),
        ]
    )

    report = drift_report(baseline, weird)

    assert report["available"] is True
    assert report["overall_status"] == "alert"
    assert report["components"]["vocabulary_oov_rate"]["status"] == "alert"
    # 5 queries is below the PSI sample floor: honesty about statistical power
    assert report["components"]["retrieval_score_psi"]["status"] == "insufficient_data"


def test_drift_report_psi_alerts_with_enough_shifted_queries():
    baseline = build_baseline(BASELINE_RECORDS)
    shifted = make_records(
        [(f"What are the principles for processing personal data variant {i}?", 0.02) for i in range(24)]
    )

    report = drift_report(baseline, shifted)

    assert report["components"]["retrieval_score_psi"]["status"] == "alert"
    assert report["overall_status"] == "alert"


def test_drift_report_unavailable_below_minimum_queries():
    baseline = build_baseline(BASELINE_RECORDS)
    report = drift_report(baseline, BASELINE_RECORDS[:2])
    assert report["available"] is False


def test_query_log_roundtrip_and_corrupt_line_tolerance(tmp_path):
    log_path = tmp_path / "query_log.jsonl"
    append_query_record("What is TLS?", 0.7, extra={"backend": "tfidf"}, log_path=log_path)
    append_query_record("What is DNS?", 0.6, log_path=log_path)
    log_path.open("a", encoding="utf-8").write("{not json}\n")

    records = read_query_log(log_path)

    assert [record.question for record in records] == ["What is TLS?", "What is DNS?"]
    assert records[0].top_score == 0.7


def test_run_drift_report_end_to_end(tmp_path):
    from adip.mlops.run_drift_report import main
    from adip.rag.retriever import build_index

    chunks = []
    for i, topic in enumerate(
        ["personal data protection regulation", "network protocol request methods", "investment fund diversification"]
    ):
        chunks.append(
            {
                "chunk_id": f"chunk_{i}", "document_id": f"doc_{i}", "filename": f"doc_{i}.md",
                "source_path": "/tmp/x.md", "source_type": "md", "checksum": "abc",
                "page_number": 1, "chunk_index": 0, "text": f"Content about {topic}.",
                "token_count": 4, "char_count": 30, "metadata": {},
            }
        )
    index_path = tmp_path / "index"
    build_index(chunks).save(index_path)
    golden_path = tmp_path / "golden.jsonl"
    golden_rows = [
        {"question": "What regulates personal data protection?", "expected_substrings": ["personal data"]},
        {"question": "Which protocol methods handle requests?", "expected_substrings": ["protocol"]},
        {"question": "How does fund diversification work?", "expected_substrings": ["diversification"]},
    ]
    golden_path.write_text("\n".join(json.dumps(r) for r in golden_rows) + "\n", encoding="utf-8")
    baseline_path = tmp_path / "baseline.json"

    exit_code = main(
        [
            "--rebuild-baseline",
            "--index", str(index_path), "--golden", str(golden_path),
            "--baseline", str(baseline_path), "--run-dir", str(tmp_path / "runs"),
        ]
    )
    assert exit_code == 0
    assert baseline_path.exists()

    log_path = tmp_path / "log.jsonl"
    for _ in range(3):
        append_query_record("What regulates personal data protection?", 0.6, log_path=log_path)
        append_query_record("Which protocol methods handle requests?", 0.55, log_path=log_path)
    report_path = tmp_path / "report.json"
    exit_code = main(
        [
            "--index", str(index_path), "--golden", str(golden_path),
            "--baseline", str(baseline_path), "--query-log", str(log_path),
            "--report-output", str(report_path), "--run-dir", str(tmp_path / "runs"),
        ]
    )
    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["available"] is True
    assert report["overall_status"] in {"ok", "warn"}
