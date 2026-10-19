"""Evidence coverage: does the retrieved evidence actually address the question?

The agent's verifier uses this to decide whether to search again. For a question
(or sub-question) it finds the single retrieved passage that covers the most of
the question's content terms, weighted by IDF so rare, specific terms ("DNS",
"idempotent", "CertificateVerify") count far more than common ones ("data",
"use").

A follow-up search is triggered only when coverage is below the threshold *and*
something specific is missing: a term that is rare in the corpus, or a named
entity from the question ("OAuth", "ImageNet", "GDPR"). The follow-up query is
every missing term, so it reads like the uncovered part of the question. A
question whose best passage merely uses different wording is missing only
common words and triggers nothing.

``DEFAULT_COVERAGE_THRESHOLD`` was set by a rule fixed in advance: the 10th
percentile of best-passage coverage over the 67 single-hop questions
(answerable golden questions plus paraphrase probes; 0.31 on the TF-IDF index),
so at most ~10% of ordinary questions pay for a second search. The multi-hop
probes were not used to choose it.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from adip.rag.rewrite import QUESTION_BOILERPLATE

TERM_PATTERN = re.compile(r"[a-z0-9]+")
# Question scaffolding that says what kind of answer is wanted, not what it is about.
EXTRA_STOPWORDS = {
    "according",
    "allowed",
    "based",
    "called",
    "does",
    "doing",
    "kind",
    "main",
    "make",
    "makes",
    "mean",
    "means",
    "much",
    "need",
    "needed",
    "say",
    "says",
    "sense",
    "type",
    "types",
    "used",
    "uses",
    "using",
    "way",
}
STOP_TERMS = frozenset(ENGLISH_STOP_WORDS) | frozenset(QUESTION_BOILERPLATE) | frozenset(EXTRA_STOPWORDS)

DEFAULT_COVERAGE_THRESHOLD = 0.3
# A missing term is "distinctive" when it appears in at most this share of chunks.
DEFAULT_DISTINCTIVE_DF_FRACTION = 0.05
MAX_FOLLOW_UP_TERMS = 8
# Capitalized or mixed-case tokens in the question: acronyms and names ("TLS", "OAuth", "Adam").
ENTITY_PATTERN = re.compile(r"\b(?:[A-Z][A-Za-z0-9]*[A-Z0-9][A-Za-z0-9]*|[A-Z][a-z]{2,})\b")
# A follow-up search is kept only if its best passage covers this share of the missing terms.
FOLLOW_UP_ACCEPT_COVERAGE = 0.5


def normalize_term(token: str) -> str:
    """Light plural folding so "tokens" matches "token" and "policies" matches "policy"."""
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def entity_terms(question: str) -> set[str]:
    """Named things in the question: acronyms, mixed-case names, and capitalized words
    that do not start a sentence."""
    entities: set[str] = set()
    for match in ENTITY_PATTERN.finditer(question):
        prefix = question[: match.start()].rstrip()
        token = match.group(0)
        sentence_start = not prefix or prefix.endswith((".", "?", "!", ":"))
        if sentence_start and token[1:].islower():
            continue
        lowered = token.lower()
        if len(lowered) >= 3 and lowered not in STOP_TERMS:
            entities.add(normalize_term(lowered))
    return entities


def content_terms(text: str) -> set[str]:
    return {
        normalize_term(token)
        for token in TERM_PATTERN.findall(str(text).lower())
        if len(token) >= 3 and token not in STOP_TERMS
    }


@dataclass(frozen=True)
class TermStatistics:
    document_frequency: dict[str, int]
    chunk_count: int

    def idf(self, term: str) -> float:
        return math.log((self.chunk_count + 1) / (self.document_frequency.get(term, 0) + 1)) + 1.0

    def is_distinctive(self, term: str, max_df_fraction: float = DEFAULT_DISTINCTIVE_DF_FRACTION) -> bool:
        return self.document_frequency.get(term, 0) <= max(1.0, max_df_fraction * self.chunk_count)


def build_term_statistics(texts: Iterable[str]) -> TermStatistics:
    frequency: Counter[str] = Counter()
    count = 0
    for text in texts:
        frequency.update(content_terms(text))
        count += 1
    return TermStatistics(document_frequency=dict(frequency), chunk_count=count)


def term_statistics_for(index: Any) -> TermStatistics:
    """Corpus term statistics for an index, computed once and cached on the index object."""
    cached = getattr(index, "_coverage_term_statistics", None)
    if cached is None or cached.chunk_count != len(index.chunks):
        cached = build_term_statistics(chunk["text"] for chunk in index.chunks)
        index._coverage_term_statistics = cached
    return cached


def weighted_coverage(query_terms: set[str], covered_terms: set[str], stats: TermStatistics) -> float:
    # fsum, because a plain sum over a set depends on its iteration order, which
    # changes with each process's string-hash seed: equal coverages must compare equal.
    total = math.fsum(stats.idf(term) for term in query_terms)
    if total <= 0:
        return 1.0
    return math.fsum(stats.idf(term) for term in query_terms & covered_terms) / total


@dataclass(frozen=True)
class CoverageResult:
    coverage: float
    best_chunk_id: str | None
    missing_terms: list[str]
    # Missing terms that justify searching again (rare in the corpus, or named in the question).
    specific_missing_terms: list[str]

    @property
    def follow_up_terms(self) -> list[str]:
        """The follow-up query: every missing term, most specific first (empty if none is specific)."""
        return self.missing_terms[:MAX_FOLLOW_UP_TERMS] if self.specific_missing_terms else []

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage": round(self.coverage, 4),
            "best_chunk_id": self.best_chunk_id,
            "missing_terms": self.missing_terms,
            "specific_missing_terms": self.specific_missing_terms,
            "follow_up_terms": self.follow_up_terms,
        }


def assess_coverage(
    question: str,
    candidate_chunks: list[dict[str, Any]],
    stats: TermStatistics,
    extra_chunks: Iterable[dict[str, Any]] = (),
    max_df_fraction: float = DEFAULT_DISTINCTIVE_DF_FRACTION,
) -> CoverageResult:
    """Coverage of ``question`` by its best single passage plus any ``extra_chunks``.

    ``extra_chunks`` are passages that earlier follow-up searches found for the
    missing terms; together with the best passage they form the evidence that
    jointly answers the question.
    """
    query_terms = content_terms(question)
    if not query_terms:
        return CoverageResult(1.0, None, [], [])
    best_chunk_id = None
    best_terms: set[str] = set()
    best_score = -1.0
    for chunk in candidate_chunks:
        terms = content_terms(chunk["text"])
        score = weighted_coverage(query_terms, terms, stats)
        if score > best_score:
            best_score, best_terms, best_chunk_id = score, terms, str(chunk.get("chunk_id"))
    covered = set(best_terms)
    for chunk in extra_chunks:
        covered |= content_terms(chunk["text"])
    coverage = weighted_coverage(query_terms, covered, stats)
    entities = entity_terms(question)
    missing = sorted(query_terms - covered, key=lambda term: (term not in entities, -stats.idf(term), term))
    specific = [term for term in missing if term in entities or stats.is_distinctive(term, max_df_fraction)]
    return CoverageResult(coverage, best_chunk_id, missing, specific)

