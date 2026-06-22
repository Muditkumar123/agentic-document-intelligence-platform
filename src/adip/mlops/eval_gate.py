"""Evaluation quality gate.

Reads the metric JSON files emitted by the tracked eval commands
(``run_rag_eval`` for retrieval, ``run_generation_eval`` for answer quality) and
compares each metric against the floors (``min``) and ceilings (``max``) declared
in a thresholds config. Prints a readable report and exits non-zero if any metric
is out of bounds, so it can gate a CI pipeline against retrieval/generation
regressions.

The deterministic extractive baseline makes both eval commands reproducible, so
these gates are stable in CI rather than flaky.

Usage::

    python -m adip.mlops.eval_gate --thresholds ci/eval_thresholds.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class GateResult:
    """Outcome of checking one metric against one bound."""

    group: str
    metric: str
    bound: str  # "min" or "max"
    threshold: float
    value: float | None
    passed: bool

    @property
    def status(self) -> str:
        if self.value is None:
            return "MISSING"
        return "PASS" if self.passed else "FAIL"


def evaluate_group(group: str, bound: str, thresholds: dict[str, float], metrics: dict[str, Any]) -> list[GateResult]:
    """Check a flat ``metric -> threshold`` map under a single bound (min/max).

    A metric that is absent or non-numeric in ``metrics`` is treated as a failure
    (``MISSING``) rather than silently passing, so a renamed or dropped metric
    cannot quietly disable a gate.
    """
    if bound not in {"min", "max"}:
        raise ValueError(f"bound must be 'min' or 'max', got {bound!r}")

    results: list[GateResult] = []
    for metric, threshold in thresholds.items():
        raw = metrics.get(metric)
        value = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None
        if value is None:
            passed = False
        elif bound == "min":
            passed = value >= threshold
        else:
            passed = value <= threshold
        results.append(GateResult(group, metric, bound, float(threshold), value, passed))
    return results


def check_thresholds(
    thresholds_config: dict[str, Any],
    *,
    base_dir: Path,
    metrics_overrides: dict[str, dict[str, Any]] | None = None,
) -> list[GateResult]:
    """Evaluate every group in ``thresholds_config`` against its metrics file.

    ``metrics_overrides`` lets callers (tests) inject metric dicts directly
    instead of reading from disk, keyed by group name.
    """
    metrics_overrides = metrics_overrides or {}
    results: list[GateResult] = []
    for group, spec in thresholds_config.items():
        if not isinstance(spec, dict):
            continue
        if group in metrics_overrides:
            metrics = metrics_overrides[group]
        else:
            metrics_path = base_dir / spec["metrics_file"]
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        results.extend(evaluate_group(group, "min", spec.get("min", {}), metrics))
        results.extend(evaluate_group(group, "max", spec.get("max", {}), metrics))
    return results


def format_report(results: list[GateResult]) -> str:
    """Render the gate results as a fixed-width table."""
    header = ("GROUP", "METRIC", "BOUND", "THRESHOLD", "VALUE", "STATUS")
    rows = [header]
    for r in results:
        value = "-" if r.value is None else f"{r.value:.4f}"
        comparator = ">=" if r.bound == "min" else "<="
        rows.append((r.group, r.metric, f"{comparator}{r.threshold:g}", f"{r.threshold:.4f}", value, r.status))

    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = []
    for idx, row in enumerate(rows):
        line = "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        lines.append(line.rstrip())
        if idx == 0:
            lines.append("  ".join("-" * widths[i] for i in range(len(header))))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adip.mlops.eval_gate",
        description="Fail when retrieval or generation eval metrics fall outside their thresholds.",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path("ci/eval_thresholds.json"),
        help="Thresholds config (groups of min/max bounds and per-group metrics_file).",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Directory the per-group metrics_file paths are resolved against.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    thresholds_config = json.loads(args.thresholds.read_text(encoding="utf-8"))
    results = check_thresholds(thresholds_config, base_dir=args.base_dir)

    print(format_report(results))
    failures = [r for r in results if not r.passed]
    print()
    if failures:
        print(f"FAILED: {len(failures)} of {len(results)} eval gate checks regressed.")
        for r in failures:
            value = "missing" if r.value is None else f"{r.value:.4f}"
            comparator = ">=" if r.bound == "min" else "<="
            print(f"  - {r.group}.{r.metric}: {value} violates {comparator} {r.threshold:g}")
        return 1
    print(f"PASSED: all {len(results)} eval gate checks within thresholds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
