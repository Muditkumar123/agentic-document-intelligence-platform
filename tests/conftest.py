"""Shared test fixtures.

The API's query-logging hook appends to the repo's real query log by default;
without isolation, every pytest run would pollute the drift monitoring data
with synthetic test questions. This autouse fixture points the log (and the
drift baseline lookup) at a per-test temporary directory instead. Tests that
need a specific path still override it explicitly via monkeypatch.
"""

import pytest


@pytest.fixture(autouse=True)
def isolate_monitoring_files(tmp_path, monkeypatch):
    import adip.monitoring.drift as drift_module

    monkeypatch.setattr(drift_module, "DEFAULT_QUERY_LOG", tmp_path / "test_query_log.jsonl")
    monkeypatch.setattr(drift_module, "DEFAULT_BASELINE", tmp_path / "test_drift_baseline.json")
