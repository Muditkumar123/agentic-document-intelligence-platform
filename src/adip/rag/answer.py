"""Baseline cited-answer generation from retrieved chunks."""

from __future__ import annotations

from adip.rag.retriever import RetrievedChunk


def build_extractive_answer(question: str, retrieved: list[RetrievedChunk]) -> str:
    """Create a transparent non-LLM answer from retrieved evidence."""
    if not retrieved:
        return (
            "I could not find relevant evidence in the indexed documents. "
            "Try indexing more documents or broadening the question."
        )

    evidence_lines = []
    for index, item in enumerate(retrieved, start=1):
        citation = item.citation_label
        snippet = item.snippet(max_chars=360)
        evidence_lines.append(f"[{index}] {snippet} ({citation})")

    evidence = "\n".join(evidence_lines)
    return (
        f"Baseline retrieval answer for: {question}\n\n"
        "The most relevant evidence I found is:\n"
        f"{evidence}\n\n"
        "This is an extractive baseline, not an LLM-generated synthesis yet."
    )
