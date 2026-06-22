"""Show RAG command hints with `python -m adip.rag`."""

from __future__ import annotations


def main() -> int:
    print("Use `python -m adip.rag.index`, `python -m adip.rag.query`, or `python -m adip.rag.evaluate`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
