"""Okapi BM25 index for the lexical side of hybrid retrieval.

Built on scikit-learn's ``CountVectorizer`` so it needs no dependencies beyond
what the TF-IDF baseline already uses, and stays deterministic (CI-safe).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75


@dataclass
class Bm25Index:
    """Term statistics needed to score documents against a query with Okapi BM25."""

    vectorizer: CountVectorizer
    matrix: object  # sparse (n_docs, n_terms) raw term counts
    idf: np.ndarray
    doc_lengths: np.ndarray
    avg_doc_length: float
    k1: float = DEFAULT_BM25_K1
    b: float = DEFAULT_BM25_B

    def scores(self, question: str) -> np.ndarray:
        """BM25 score of every indexed document for ``question`` (0 when no term overlap)."""
        query_counts = self.vectorizer.transform([question])
        term_indices = query_counts.indices
        n_docs = self.matrix.shape[0]
        if term_indices.size == 0:
            return np.zeros(n_docs, dtype="float64")

        # (n_docs, n_query_terms) term frequencies; query term counts are few,
        # so densifying this slice is cheap even for large corpora.
        term_frequencies = np.asarray(self.matrix[:, term_indices].todense(), dtype="float64")
        length_norm = self.k1 * (
            1.0 - self.b + self.b * (self.doc_lengths / max(self.avg_doc_length, 1e-9))
        )
        numerator = term_frequencies * (self.k1 + 1.0)
        denominator = term_frequencies + length_norm[:, np.newaxis]
        return (self.idf[term_indices][np.newaxis, :] * (numerator / denominator)).sum(axis=1)


def build_bm25_index(
    texts: list[str],
    k1: float = DEFAULT_BM25_K1,
    b: float = DEFAULT_BM25_B,
) -> Bm25Index:
    if k1 <= 0:
        raise ValueError("k1 must be greater than 0")
    if not 0.0 <= b <= 1.0:
        raise ValueError("b must be between 0 and 1")

    vectorizer = CountVectorizer(lowercase=True, stop_words="english")
    matrix = vectorizer.fit_transform(texts).tocsc()
    n_docs = matrix.shape[0]
    document_frequencies = np.asarray((matrix > 0).sum(axis=0)).ravel().astype("float64")
    idf = np.log(1.0 + (n_docs - document_frequencies + 0.5) / (document_frequencies + 0.5))
    doc_lengths = np.asarray(matrix.sum(axis=1)).ravel().astype("float64")
    avg_doc_length = float(doc_lengths.mean()) if n_docs else 0.0
    return Bm25Index(
        vectorizer=vectorizer,
        matrix=matrix,
        idf=idf,
        doc_lengths=doc_lengths,
        avg_doc_length=avg_doc_length,
        k1=k1,
        b=b,
    )
