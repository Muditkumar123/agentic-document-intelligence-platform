"""Query rewriting / expansion before retrieval.

Retrieval is only as good as the query: lexical backends miss paraphrases and
morphological variants ("principles" never matches "principle" in TF-IDF), and
users rarely phrase questions in the corpus's vocabulary. This module rewrites
the question into a small set of variants, retrieves for every variant, and
fuses the ranked lists with reciprocal-rank fusion — the same rank-based, no
score-calibration trick the hybrid backend uses.

Two modes behind one interface:

- ``keywords`` — deterministic expansion (content-term query + singular/plural
  variants), CI-safe, no dependencies.
- ``llm`` — an OpenAI-compatible model generates paraphrases from the versioned
  ``rewrite`` prompt; opt-in and offline like the judge, with a callable
  protocol so tests inject fakes. A rewriter failure degrades to the original
  question instead of failing retrieval.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

from adip.rag.retriever import RagIndex, RetrievedChunk

SUPPORTED_REWRITERS = {"none", "keywords", "llm"}
DEFAULT_REWRITE_COUNT = 3
DEFAULT_REWRITE_RRF_K = 60
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")
QUESTION_BOILERPLATE = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "can",
    "could",
    "define",
    "describe",
    "do",
    "does",
    "explain",
    "for",
    "from",
    "give",
    "how",
    "in",
    "is",
    "it",
    "list",
    "me",
    "of",
    "on",
    "please",
    "should",
    "tell",
    "the",
    "to",
    "under",
    "us",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
}


@runtime_checkable
class QueryRewriter(Protocol):
    """Produces alternative phrasings for a question (empty list on failure)."""

    def __call__(self, question: str) -> list[str]: ...


def validate_rewriter(rewriter: str) -> None:
    if rewriter not in SUPPORTED_REWRITERS:
        raise ValueError(f"Unsupported query rewriter: {rewriter}")


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _morphological_variant(token: str) -> str | None:
    """Cheap singular/plural counterpart of a content token (None when unsure)."""
    if len(token) < 4:
        return None
    if token.endswith("ies"):
        return f"{token[:-3]}y"
    if token.endswith("ses") or token.endswith("xes") or token.endswith("hes"):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return f"{token}s"


def expand_query_keywords(question: str) -> list[str]:
    """Deterministic variants: the original, a content-term query, and a
    morphological-variant query. Duplicates and empty variants are dropped."""
    content_terms = [token for token in _tokens(question) if token not in QUESTION_BOILERPLATE]
    variants = [question]
    if content_terms:
        variants.append(" ".join(content_terms))
        morphed = [
            _morphological_variant(token) or token
            for token in content_terms
        ]
        variants.append(" ".join(morphed))

    seen: set[str] = set()
    unique: list[str] = []
    for variant in variants:
        normalized = variant.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(variant.strip())
    return unique


def fuse_ranked_lists(
    ranked_lists: list[list[RetrievedChunk]],
    rrf_k: int = DEFAULT_REWRITE_RRF_K,
) -> list[RetrievedChunk]:
    """Weighted-equal reciprocal-rank fusion of per-variant result lists.

    Chunks are keyed by ``chunk_id``; each list contributes ``1 / (rrf_k + rank)``
    per chunk. Scores are normalized so a chunk ranked first in every list gets
    1.0, keeping downstream abstention thresholds meaningful.
    """
    if rrf_k < 1:
        raise ValueError("rrf_k must be greater than or equal to 1")
    non_empty = [items for items in ranked_lists if items]
    if not non_empty:
        return []

    fused: dict[str, float] = {}
    chunks: dict[str, dict[str, Any]] = {}
    for items in non_empty:
        for item in items:
            chunk_id = str(item.chunk.get("chunk_id"))
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + item.rank)
            chunks.setdefault(chunk_id, item.chunk)

    normalizer = (rrf_k + 1) / len(non_empty)
    ordered = sorted(fused.items(), key=lambda entry: entry[1], reverse=True)
    return [
        RetrievedChunk(chunk=chunks[chunk_id], score=score * normalizer, rank=rank)
        for rank, (chunk_id, score) in enumerate(ordered, start=1)
    ]


def rewrite_question(
    question: str,
    rewriter: str,
    llm_rewriter: QueryRewriter | None = None,
    rewrite_count: int = DEFAULT_REWRITE_COUNT,
) -> list[str]:
    """All query variants to retrieve for, always starting with the original."""
    validate_rewriter(rewriter)
    if rewriter == "none":
        return [question]
    if rewriter == "keywords":
        return expand_query_keywords(question)

    if llm_rewriter is None:
        raise ValueError("llm rewriter mode requires an LLM query rewriter")
    variants = [question]
    for rewrite in llm_rewriter(question)[:rewrite_count]:
        cleaned = rewrite.strip()
        if cleaned and cleaned.lower() not in {v.lower() for v in variants}:
            variants.append(cleaned)
    return variants


def retrieve_with_rewrites(
    index: RagIndex,
    variants: list[str],
    top_k: int,
    document_filter: str | None = None,
    rrf_k: int = DEFAULT_REWRITE_RRF_K,
) -> list[RetrievedChunk]:
    """Retrieve for every variant and fuse the ranked lists (single-variant
    input short-circuits to a plain search)."""
    if not variants:
        raise ValueError("At least one query variant is required")
    if len(variants) == 1:
        return index.search(variants[0], top_k=top_k, document_filter=document_filter)

    ranked_lists = [
        index.search(variant, top_k=top_k, document_filter=document_filter)
        for variant in variants
    ]
    return fuse_ranked_lists(ranked_lists, rrf_k=rrf_k)[:top_k]


def parse_rewrites(text: str) -> list[str]:
    """Extract paraphrases from LLM output: one per line, tolerating numbering,
    bullets, and surrounding prose."""
    rewrites: list[str] = []
    for line in (text or "").splitlines():
        cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"')
        if cleaned and len(cleaned.split()) >= 3 and not cleaned.endswith(":"):
            rewrites.append(cleaned)
    return rewrites


class LLMQueryRewriter:
    """Generates paraphrases with an OpenAI-compatible model using the versioned
    rewrite prompt. Failures return an empty list so retrieval falls back to the
    original question."""

    def __init__(
        self,
        model_name: str,
        endpoint_url: str | None = None,
        api_key: str | None = None,
        rewrite_count: int = DEFAULT_REWRITE_COUNT,
        max_new_tokens: int = 256,
    ) -> None:
        from adip.llmops.models import OpenAICompatibleChatAdapter
        from adip.llmops.prompts import load_prompt_template

        self.prompt = load_prompt_template("rewrite")
        self.rewrite_count = rewrite_count
        self.max_new_tokens = max_new_tokens
        self.adapter = OpenAICompatibleChatAdapter(
            model_name=model_name,
            endpoint_url=endpoint_url,
            api_key=api_key,
        )

    def __call__(self, question: str) -> list[str]:
        from adip.llmops.models import GenerationRequest, strip_reasoning_blocks

        rendered = self.prompt.render(question=question, rewrite_count=self.rewrite_count)
        try:
            response = self.adapter.generate(
                GenerationRequest(
                    prompt=rendered,
                    question=question,
                    task_type="rewrite",
                    domain_preset="general",
                    evidence=[],
                    max_new_tokens=self.max_new_tokens,
                )
            )
        except RuntimeError as exc:
            print(f"query rewriter failed for {question[:60]!r}: {exc}")
            return []
        return parse_rewrites(strip_reasoning_blocks(response.text))[: self.rewrite_count]
