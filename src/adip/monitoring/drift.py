"""Text and retrieval-score drift detection, dependency-light and deterministic.

The baseline is built from the golden questions — the exact distribution every
quality number was measured on. Incoming queries are compared against it along
three axes:

- **vocabulary** (out-of-vocabulary token rate): are users asking about things
  the corpus was never validated for?
- **question length** (z-score of the recent mean against the baseline);
- **retrieval confidence** (PSI — population stability index — over the top
  retrieval score, binned on baseline deciles). PSI conventions: < 0.1 stable,
  0.1–0.25 moderate shift (warn), > 0.25 significant shift (alert).

Everything is plain Python + math on JSON records, so drift reports are exactly
as reproducible as the rest of the monitoring stack.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")
DEFAULT_QUERY_LOG = Path("data/monitoring/query_log.jsonl")
DEFAULT_BASELINE = Path("data/monitoring/drift_baseline.json")
PSI_WARN = 0.1
PSI_ALERT = 0.25
OOV_WARN = 0.35
OOV_ALERT = 0.6
LENGTH_Z_WARN = 2.0
LENGTH_Z_ALERT = 3.0
MIN_QUERIES_FOR_REPORT = 5
# PSI over binned scores is statistically meaningless on tiny samples: a handful
# of queries spread over decile bins produces large PSI values by chance alone.
PSI_MIN_QUERIES = 20


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


@dataclass(frozen=True)
class QueryRecord:
    question: str
    top_score: float


def build_baseline(records: list[QueryRecord]) -> dict[str, Any]:
    """Summarize the reference distribution (typically the golden questions)."""
    if len(records) < 2:
        raise ValueError("At least 2 reference queries are required for a baseline")

    vocabulary: Counter[str] = Counter()
    lengths: list[int] = []
    scores: list[float] = []
    for record in records:
        tokens = tokenize(record.question)
        vocabulary.update(tokens)
        lengths.append(len(tokens))
        scores.append(float(record.top_score))

    length_mean = sum(lengths) / len(lengths)
    length_var = sum((value - length_mean) ** 2 for value in lengths) / len(lengths)
    sorted_scores = sorted(scores)
    decile_edges = [
        sorted_scores[min(len(sorted_scores) - 1, int(len(sorted_scores) * fraction))]
        for fraction in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    ]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query_count": len(records),
        "vocabulary": dict(vocabulary),
        "length_mean": length_mean,
        "length_std": math.sqrt(length_var),
        "score_bin_edges": decile_edges,
        "score_bin_proportions": _bin_proportions(scores, decile_edges),
        "score_mean": sum(scores) / len(scores),
    }


def _bin_proportions(values: list[float], edges: list[float]) -> list[float]:
    counts = [0] * (len(edges) + 1)
    for value in values:
        position = 0
        while position < len(edges) and value > edges[position]:
            position += 1
        counts[position] += 1
    total = max(1, len(values))
    return [count / total for count in counts]


def population_stability_index(expected: list[float], actual: list[float]) -> float:
    """PSI with epsilon smoothing so empty bins cannot divide by zero."""
    if len(expected) != len(actual):
        raise ValueError("expected and actual bin proportions must have the same length")
    epsilon = 1e-4
    psi = 0.0
    for expected_share, actual_share in zip(expected, actual):
        e = max(expected_share, epsilon)
        a = max(actual_share, epsilon)
        psi += (a - e) * math.log(a / e)
    return psi


def _grade(value: float, warn: float, alert: float) -> str:
    if value >= alert:
        return "alert"
    if value >= warn:
        return "warn"
    return "ok"


def drift_report(baseline: dict[str, Any], recent: list[QueryRecord]) -> dict[str, Any]:
    """Compare recent queries against the baseline along all three axes."""
    if len(recent) < MIN_QUERIES_FOR_REPORT:
        return {
            "available": False,
            "reason": f"Need at least {MIN_QUERIES_FOR_REPORT} logged queries, have {len(recent)}",
            "recent_query_count": len(recent),
        }

    vocabulary = set(baseline["vocabulary"])
    oov_rates: list[float] = []
    lengths: list[int] = []
    scores: list[float] = []
    for record in recent:
        tokens = tokenize(record.question)
        if tokens:
            oov_rates.append(sum(1 for token in tokens if token not in vocabulary) / len(tokens))
            lengths.append(len(tokens))
        scores.append(float(record.top_score))

    oov_rate = sum(oov_rates) / len(oov_rates) if oov_rates else 0.0
    length_mean = sum(lengths) / len(lengths) if lengths else 0.0
    length_std = max(baseline["length_std"], 1e-6)
    length_z = abs(length_mean - baseline["length_mean"]) / length_std
    score_psi = population_stability_index(
        baseline["score_bin_proportions"],
        _bin_proportions(scores, baseline["score_bin_edges"]),
    )
    psi_status = (
        _grade(score_psi, PSI_WARN, PSI_ALERT)
        if len(scores) >= PSI_MIN_QUERIES
        else "insufficient_data"
    )

    components = {
        "vocabulary_oov_rate": {
            "value": oov_rate,
            "status": _grade(oov_rate, OOV_WARN, OOV_ALERT),
            "meaning": "share of query tokens never seen in the reference questions",
        },
        "question_length_z": {
            "value": length_z,
            "status": _grade(length_z, LENGTH_Z_WARN, LENGTH_Z_ALERT),
            "meaning": "how far the recent mean question length sits from the baseline",
        },
        "retrieval_score_psi": {
            "value": score_psi,
            "status": psi_status,
            "meaning": (
                "population stability index of top retrieval scores over baseline deciles"
                if psi_status != "insufficient_data"
                else f"PSI needs at least {PSI_MIN_QUERIES} queries to be meaningful (have {len(scores)})"
            ),
        },
    }
    order = {"insufficient_data": 0, "ok": 0, "warn": 1, "alert": 2}
    overall = max((component["status"] for component in components.values()), key=order.get)
    return {
        "available": True,
        "overall_status": overall,
        "recent_query_count": len(recent),
        "baseline_query_count": baseline["query_count"],
        "baseline_created_at": baseline.get("created_at"),
        "recent_score_mean": sum(scores) / len(scores) if scores else None,
        "baseline_score_mean": baseline.get("score_mean"),
        "components": components,
    }


def append_query_record(
    question: str,
    top_score: float,
    extra: dict[str, Any] | None = None,
    log_path: Path | None = None,
) -> None:
    """Best-effort append to the query log; monitoring must never break queries."""
    try:
        log_path = log_path if log_path is not None else DEFAULT_QUERY_LOG
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "top_score": float(top_score),
            **(extra or {}),
        }
        with log_path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError:
        pass


def read_query_log(log_path: Path | None = None, limit: int = 500) -> list[QueryRecord]:
    log_path = log_path if log_path is not None else DEFAULT_QUERY_LOG
    if not log_path.exists():
        return []
    records: list[QueryRecord] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            records.append(
                QueryRecord(
                    question=str(payload["question"]),
                    top_score=float(payload.get("top_score", 0.0)),
                )
            )
        except (ValueError, KeyError):
            continue
    return records[-limit:]


def load_baseline(baseline_path: Path | None = None) -> dict[str, Any] | None:
    baseline_path = baseline_path if baseline_path is not None else DEFAULT_BASELINE
    if not baseline_path.exists():
        return None
    try:
        return json.loads(baseline_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
