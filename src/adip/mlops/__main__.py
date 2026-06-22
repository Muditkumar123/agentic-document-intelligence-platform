"""Show MLOps command hints with `python -m adip.mlops`."""

from __future__ import annotations


def main() -> int:
    print(
        "Use `python -m adip.mlops.run_ingestion`, "
        "`python -m adip.mlops.run_rag_eval`, or "
        "`python -m adip.mlops.run_retrieval_benchmark`, or "
        "`python -m adip.mlops.run_llmops_smoke`, or "
        "`python -m adip.mlops.run_agent`."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
