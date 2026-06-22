"""Tests for the CI evaluation quality gate."""

from __future__ import annotations

import json

from adip.mlops.eval_gate import check_thresholds, evaluate_group, format_report, main


def test_min_bound_passes_when_value_meets_floor():
    results = evaluate_group("retrieval", "min", {"mrr": 0.65}, {"mrr": 0.7667})
    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].status == "PASS"


def test_min_bound_fails_when_value_below_floor():
    results = evaluate_group("generation", "min", {"gen_eval_mean_faithfulness": 0.55}, {"gen_eval_mean_faithfulness": 0.40})
    assert results[0].passed is False
    assert results[0].status == "FAIL"


def test_max_bound_fails_when_value_above_ceiling():
    results = evaluate_group("generation", "max", {"gen_eval_refusal_rate": 0.2}, {"gen_eval_refusal_rate": 0.5})
    assert results[0].passed is False
    assert results[0].bound == "max"


def test_max_bound_passes_when_value_within_ceiling():
    results = evaluate_group("generation", "max", {"gen_eval_refusal_rate": 0.2}, {"gen_eval_refusal_rate": 0.0})
    assert results[0].passed is True


def test_missing_metric_is_treated_as_failure():
    results = evaluate_group("retrieval", "min", {"hit_rate_at_k": 0.85}, {"some_other_metric": 1.0})
    assert results[0].passed is False
    assert results[0].status == "MISSING"
    assert results[0].value is None


def test_boolean_value_is_not_accepted_as_numeric():
    # bool is a subclass of int; it must not be mistaken for a real metric value.
    results = evaluate_group("generation", "min", {"grounded": 0.5}, {"grounded": True})
    assert results[0].status == "MISSING"


def test_check_thresholds_with_overrides_aggregates_groups():
    config = {
        "retrieval": {"metrics_file": "unused.json", "min": {"mrr": 0.65}},
        "generation": {
            "metrics_file": "unused.json",
            "min": {"gen_eval_mean_faithfulness": 0.55},
            "max": {"gen_eval_refusal_rate": 0.2},
        },
    }
    overrides = {
        "retrieval": {"mrr": 0.70},
        "generation": {"gen_eval_mean_faithfulness": 0.60, "gen_eval_refusal_rate": 0.0},
    }
    results = check_thresholds(config, base_dir=None, metrics_overrides=overrides)  # type: ignore[arg-type]
    assert len(results) == 3
    assert all(r.passed for r in results)


def test_format_report_contains_status_column():
    results = evaluate_group("retrieval", "min", {"mrr": 0.65}, {"mrr": 0.7667})
    report = format_report(results)
    assert "STATUS" in report
    assert "PASS" in report
    assert "mrr" in report


def test_main_passes_on_good_metrics(tmp_path, capsys):
    (tmp_path / "data" / "monitoring").mkdir(parents=True)
    (tmp_path / "data" / "monitoring" / "rag.json").write_text(
        json.dumps({"hit_rate_at_k": 1.0, "mrr": 0.7667}), encoding="utf-8"
    )
    (tmp_path / "data" / "monitoring" / "gen.json").write_text(
        json.dumps(
            {
                "gen_eval_mean_faithfulness": 0.6787,
                "gen_eval_grounded_rate": 1.0,
                "gen_eval_mean_expected_coverage": 0.9,
                "gen_eval_mean_answer_relevance": 1.0,
                "gen_eval_mean_citation_coverage": 0.61,
                "gen_eval_refusal_rate": 0.0,
            }
        ),
        encoding="utf-8",
    )
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(
        json.dumps(
            {
                "retrieval": {"metrics_file": "data/monitoring/rag.json", "min": {"hit_rate_at_k": 0.85, "mrr": 0.65}},
                "generation": {
                    "metrics_file": "data/monitoring/gen.json",
                    "min": {"gen_eval_mean_faithfulness": 0.55, "gen_eval_grounded_rate": 0.8},
                    "max": {"gen_eval_refusal_rate": 0.2},
                },
            }
        ),
        encoding="utf-8",
    )
    exit_code = main(["--thresholds", str(thresholds), "--base-dir", str(tmp_path)])
    assert exit_code == 0
    assert "PASSED" in capsys.readouterr().out


def test_main_fails_and_reports_regression(tmp_path, capsys):
    (tmp_path / "data" / "monitoring").mkdir(parents=True)
    (tmp_path / "data" / "monitoring" / "rag.json").write_text(
        json.dumps({"hit_rate_at_k": 0.40, "mrr": 0.30}), encoding="utf-8"
    )
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(
        json.dumps(
            {"retrieval": {"metrics_file": "data/monitoring/rag.json", "min": {"hit_rate_at_k": 0.85, "mrr": 0.65}}}
        ),
        encoding="utf-8",
    )
    exit_code = main(["--thresholds", str(thresholds), "--base-dir", str(tmp_path)])
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "FAILED" in out
    assert "hit_rate_at_k" in out
