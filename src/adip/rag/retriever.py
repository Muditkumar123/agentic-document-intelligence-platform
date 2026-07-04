"""Local retrieval indexes for RAG."""

from __future__ import annotations

import importlib.util
import pickle
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize

from adip.rag.bm25 import DEFAULT_BM25_B, DEFAULT_BM25_K1, Bm25Index, build_bm25_index

INDEX_FILENAME = "rag_index.pkl"
FAISS_INDEX_FILENAME = "faiss.index"
DENSE_BACKENDS = {"dense", "dense_lsa", "sentence_transformers"}
HYBRID_BACKEND = "hybrid"
DEFAULT_RRF_K = 60
DEFAULT_HYBRID_DENSE_WEIGHT = 0.5
GENERIC_DOCUMENT_NOUNS = {
    "doc",
    "document",
    "file",
    "markdown",
    "md",
    "paper",
    "pdf",
    "text",
    "upload",
    "uploaded",
}
GENERIC_SUMMARY_TERMS = {
    "about",
    "brief",
    "overview",
    "summarise",
    "summarize",
    "summary",
}
GENERIC_DOCUMENT_PHRASES = (
    "this document",
    "this file",
    "this md",
    "this paper",
    "this pdf",
    "the document",
    "the file",
    "the paper",
    "the pdf",
    "uploaded document",
    "uploaded file",
    "uploaded paper",
    "uploaded pdf",
    "what is this",
    "what this",
)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: dict[str, Any]
    score: float
    rank: int

    @property
    def citation_label(self) -> str:
        page_number = self.chunk.get("page_number", "?")
        filename = self.chunk.get("filename", "unknown")
        chunk_id = self.chunk.get("chunk_id", "unknown")
        return f"{filename} p.{page_number} chunk {chunk_id}"

    def snippet(self, max_chars: int = 500) -> str:
        text = " ".join(str(self.chunk["text"]).split())
        if len(text) <= max_chars:
            return text
        return f"{text[: max_chars - 3].rstrip()}..."

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "citation": self.citation_label,
            "chunk": self.chunk,
        }


@dataclass
class RagIndex:
    backend: str
    vectorizer: TfidfVectorizer | None
    matrix: Any
    chunks: list[dict[str, Any]]
    svd: Any | None = None
    embedding_model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    faiss_index: Any | None = None
    bm25: Bm25Index | None = None
    rrf_k: int = DEFAULT_RRF_K
    hybrid_dense_weight: float = DEFAULT_HYBRID_DENSE_WEIGHT

    @property
    def vocabulary_size(self) -> int:
        if self.vectorizer is None:
            return 0
        return len(self.vectorizer.vocabulary_)

    def search(
        self,
        question: str,
        top_k: int = 5,
        document_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if self.backend == "tfidf":
            return self._search_tfidf(question, top_k, document_filter=document_filter)
        if self.backend in DENSE_BACKENDS:
            return self._search_dense(question, top_k, document_filter=document_filter)
        if self.backend == HYBRID_BACKEND:
            return self._search_hybrid(question, top_k, document_filter=document_filter)
        raise ValueError(f"Unsupported index backend: {self.backend}")

    def _search_tfidf(
        self,
        question: str,
        top_k: int,
        document_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        if self.vectorizer is None:
            raise ValueError("TF-IDF index is missing its vectorizer")
        candidate_indices = self._candidate_indices(document_filter)
        if not candidate_indices:
            return []

        query_vector = self.vectorizer.transform([question])
        similarities = linear_kernel(query_vector, self.matrix).ravel()
        top_indices = sorted(
            candidate_indices,
            key=lambda index: similarities[index],
            reverse=True,
        )[:top_k]

        results: list[RetrievedChunk] = []
        for rank, index in enumerate(top_indices, start=1):
            score = float(similarities[index])
            if score <= 0:
                continue
            results.append(RetrievedChunk(chunk=self.chunks[int(index)], score=score, rank=rank))
        if not results:
            fallback_chunks = [self.chunks[int(index)] for index in candidate_indices]
            return self._single_document_fallback(question, top_k, chunks=fallback_chunks)
        return results

    def _search_dense(
        self,
        question: str,
        top_k: int,
        document_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        candidate_indices = self._candidate_indices(document_filter)
        if not candidate_indices:
            return []

        query_vector = self._embed_query(question)
        if self.faiss_index is not None and document_filter is None:
            scores, indices = self.faiss_index.search(query_vector.astype("float32"), top_k)
            score_values = scores[0]
            top_indices = indices[0]
        else:
            similarities = np.matmul(self.matrix, query_vector[0])
            top_indices = sorted(
                candidate_indices,
                key=lambda index: similarities[index],
                reverse=True,
            )[:top_k]
            score_values = similarities[top_indices]

        results: list[RetrievedChunk] = []
        for rank, (index, score) in enumerate(zip(top_indices, score_values), start=1):
            if int(index) < 0:
                continue
            results.append(RetrievedChunk(chunk=self.chunks[int(index)], score=float(score), rank=rank))
        if not results:
            fallback_chunks = [self.chunks[int(index)] for index in candidate_indices]
            return self._single_document_fallback(question, top_k, chunks=fallback_chunks)
        return results

    def _search_hybrid(
        self,
        question: str,
        top_k: int,
        document_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        if self.bm25 is None:
            raise ValueError("Hybrid index is missing its BM25 component")
        candidate_indices = self._candidate_indices(document_filter)
        if not candidate_indices:
            return []

        bm25_scores = self.bm25.scores(question)
        dense_query = self._embed_query(question)
        dense_scores = np.matmul(self.matrix, dense_query[0])
        dense_weight = self.hybrid_dense_weight
        fused = reciprocal_rank_fusion(
            score_lists=[bm25_scores, dense_scores],
            weights=[1.0 - dense_weight, dense_weight],
            candidate_indices=candidate_indices,
            rrf_k=self.rrf_k,
        )

        top_indices = sorted(candidate_indices, key=lambda index: fused[index], reverse=True)[:top_k]
        results: list[RetrievedChunk] = []
        for rank, index in enumerate(top_indices, start=1):
            score = float(fused[index])
            if score <= 0:
                continue
            results.append(RetrievedChunk(chunk=self.chunks[int(index)], score=score, rank=rank))
        if not results:
            fallback_chunks = [self.chunks[int(index)] for index in candidate_indices]
            return self._single_document_fallback(question, top_k, chunks=fallback_chunks)
        return results

    def _embed_query(self, question: str) -> np.ndarray:
        if self.embedding_model and self.embedding_model != "lsa":
            return encode_with_sentence_transformers([question], self.embedding_model)
        if self.vectorizer is None:
            raise ValueError("Dense LSA index is missing its vectorizer")
        sparse_query = self.vectorizer.transform([question])
        if self.svd is not None:
            dense_query = self.svd.transform(sparse_query)
        else:
            dense_query = sparse_query.toarray()
        return normalize_dense_matrix(dense_query)

    def _single_document_fallback(
        self,
        question: str,
        top_k: int,
        chunks: list[dict[str, Any]] | None = None,
    ) -> list[RetrievedChunk]:
        if not looks_like_generic_document_request(question):
            return []
        fallback_chunks = generic_fallback_document_chunks(chunks or self.chunks, question)
        if not fallback_chunks:
            return []

        ordered_chunks = sorted(
            fallback_chunks,
            key=lambda chunk: (
                int(chunk.get("chunk_index", 0)),
                int(chunk.get("page_number", 0) or 0),
                str(chunk.get("chunk_id", "")),
            ),
        )
        return [
            RetrievedChunk(chunk=chunk, score=0.0, rank=rank)
            for rank, chunk in enumerate(ordered_chunks[:top_k], start=1)
        ]

    def _candidate_indices(self, document_filter: str | None) -> list[int]:
        if not document_filter:
            return list(range(len(self.chunks)))
        return [
            index
            for index, chunk in enumerate(self.chunks)
            if matches_document_filter(chunk, document_filter)
        ]

    def save(self, index_dir: Path) -> None:
        path = index_dir.expanduser()
        path.mkdir(parents=True, exist_ok=True)
        serializable = replace(self, faiss_index=None)
        with (path / INDEX_FILENAME).open("wb") as file_obj:
            pickle.dump(serializable, file_obj)
        if self.faiss_index is not None:
            faiss = import_optional_faiss()
            if faiss is not None:
                faiss.write_index(self.faiss_index, str(path / FAISS_INDEX_FILENAME))


def build_index(
    chunks: list[dict[str, Any]],
    backend: str = "tfidf",
    ngram_max: int = 2,
    embedding_model: str = "lsa",
    dense_dimensions: int = 128,
    use_faiss: bool = True,
    rrf_k: int = DEFAULT_RRF_K,
    hybrid_dense_weight: float = DEFAULT_HYBRID_DENSE_WEIGHT,
) -> RagIndex:
    if backend == "tfidf":
        return build_tfidf_index(chunks, ngram_max=ngram_max)
    if backend in DENSE_BACKENDS:
        return build_dense_index(
            chunks,
            embedding_model=embedding_model,
            dense_dimensions=dense_dimensions,
            use_faiss=use_faiss,
        )
    if backend == HYBRID_BACKEND:
        return build_hybrid_index(
            chunks,
            embedding_model=embedding_model,
            dense_dimensions=dense_dimensions,
            rrf_k=rrf_k,
            hybrid_dense_weight=hybrid_dense_weight,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def reciprocal_rank_fusion(
    score_lists: list[np.ndarray],
    weights: list[float],
    candidate_indices: list[int],
    rrf_k: int = DEFAULT_RRF_K,
) -> dict[int, float]:
    """Weighted reciprocal-rank fusion over the candidate set.

    Rank positions (not raw score magnitudes) drive the fusion, so BM25 and cosine
    scores need no scale calibration. Documents with a non-positive score in a
    component contribute nothing from that component, so a query with no term
    overlap cannot be promoted by its BM25 rank alone. Scores are normalized so a
    document ranked first in every component scores 1.0.
    """
    if len(score_lists) != len(weights):
        raise ValueError("score_lists and weights must have the same length")
    if rrf_k < 1:
        raise ValueError("rrf_k must be greater than or equal to 1")
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("weights must sum to a positive value")

    fused = {index: 0.0 for index in candidate_indices}
    for scores, weight in zip(score_lists, weights):
        if weight <= 0:
            continue
        ordered = sorted(candidate_indices, key=lambda index: scores[index], reverse=True)
        for rank, index in enumerate(ordered, start=1):
            if scores[index] <= 0:
                continue
            fused[index] += weight / (rrf_k + rank)
    normalizer = (rrf_k + 1) / total_weight
    return {index: score * normalizer for index, score in fused.items()}


def build_hybrid_index(
    chunks: list[dict[str, Any]],
    embedding_model: str = "lsa",
    dense_dimensions: int = 128,
    rrf_k: int = DEFAULT_RRF_K,
    hybrid_dense_weight: float = DEFAULT_HYBRID_DENSE_WEIGHT,
    bm25_k1: float = DEFAULT_BM25_K1,
    bm25_b: float = DEFAULT_BM25_B,
) -> RagIndex:
    if not 0.0 <= hybrid_dense_weight <= 1.0:
        raise ValueError("hybrid_dense_weight must be between 0 and 1")

    texts = [chunk["text"] for chunk in chunks]
    bm25 = build_bm25_index(texts, k1=bm25_k1, b=bm25_b)
    if embedding_model == "lsa":
        vectorizer, svd, matrix = build_lsa_embeddings(texts, dense_dimensions=dense_dimensions)
    else:
        vectorizer = None
        svd = None
        matrix = encode_with_sentence_transformers(texts, embedding_model)

    return RagIndex(
        backend=HYBRID_BACKEND,
        vectorizer=vectorizer,
        matrix=matrix,
        chunks=chunks,
        svd=svd,
        embedding_model=embedding_model,
        bm25=bm25,
        rrf_k=rrf_k,
        hybrid_dense_weight=hybrid_dense_weight,
        metadata={
            "retrieval_backend": HYBRID_BACKEND,
            "fusion": "weighted_rrf",
            "rrf_k": rrf_k,
            "hybrid_dense_weight": hybrid_dense_weight,
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "embedding_model": embedding_model,
            "dense_dimensions": int(matrix.shape[1]) if len(matrix.shape) == 2 else 0,
            "matrix_shape": tuple(matrix.shape),
        },
    )


def build_tfidf_index(chunks: list[dict[str, Any]], ngram_max: int = 2) -> RagIndex:
    if ngram_max < 1:
        raise ValueError("ngram_max must be greater than or equal to 1")

    texts = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, ngram_max),
        max_features=100_000,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(texts)
    return RagIndex(
        backend="tfidf",
        vectorizer=vectorizer,
        matrix=matrix,
        chunks=chunks,
        metadata={
            "retrieval_backend": "tfidf",
            "ngram_max": ngram_max,
            "matrix_shape": tuple(matrix.shape),
        },
    )


def build_dense_index(
    chunks: list[dict[str, Any]],
    embedding_model: str = "lsa",
    dense_dimensions: int = 128,
    use_faiss: bool = True,
) -> RagIndex:
    if dense_dimensions < 1:
        raise ValueError("dense_dimensions must be greater than or equal to 1")

    texts = [chunk["text"] for chunk in chunks]
    if embedding_model == "lsa":
        vectorizer, svd, matrix = build_lsa_embeddings(texts, dense_dimensions=dense_dimensions)
        backend = "dense_lsa"
    else:
        vectorizer = None
        svd = None
        matrix = encode_with_sentence_transformers(texts, embedding_model)
        backend = "sentence_transformers"

    faiss_index = build_faiss_index(matrix) if use_faiss else None
    return RagIndex(
        backend=backend,
        vectorizer=vectorizer,
        matrix=matrix,
        chunks=chunks,
        svd=svd,
        embedding_model=embedding_model,
        metadata={
            "retrieval_backend": backend,
            "embedding_model": embedding_model,
            "dense_dimensions": int(matrix.shape[1]) if len(matrix.shape) == 2 else 0,
            "faiss_enabled": faiss_index is not None,
            "faiss_requested": use_faiss,
            "matrix_shape": tuple(matrix.shape),
        },
        faiss_index=faiss_index,
    )


def build_lsa_embeddings(
    texts: list[str],
    dense_dimensions: int,
) -> tuple[TfidfVectorizer, TruncatedSVD | None, np.ndarray]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=100_000,
        norm="l2",
    )
    sparse_matrix = vectorizer.fit_transform(texts)
    max_components = max(0, min(sparse_matrix.shape) - 1)
    if max_components >= 1:
        n_components = min(dense_dimensions, max_components)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        matrix = svd.fit_transform(sparse_matrix)
    else:
        svd = None
        matrix = sparse_matrix.toarray()
    return vectorizer, svd, normalize_dense_matrix(matrix)


def encode_with_sentence_transformers(texts: list[str], model_name: str) -> np.ndarray:
    if importlib.util.find_spec("sentence_transformers") is None:
        raise ImportError(
            "sentence-transformers is required for embedding_model values other than `lsa`."
        )
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype="float32")


def normalize_dense_matrix(matrix: Any) -> np.ndarray:
    array = np.asarray(matrix, dtype="float32")
    return normalize(array, norm="l2", axis=1).astype("float32")


def looks_like_generic_document_request(question: str) -> bool:
    lowered = question.lower()
    if any(phrase in lowered for phrase in GENERIC_DOCUMENT_PHRASES):
        return True

    tokens = set(re.findall(r"[a-zA-Z0-9_]+", lowered))
    mentions_document = bool(tokens & GENERIC_DOCUMENT_NOUNS)
    asks_for_summary = bool(tokens & GENERIC_SUMMARY_TERMS) or "tell me about" in lowered
    return mentions_document and asks_for_summary


def index_has_single_document(chunks: list[dict[str, Any]]) -> bool:
    document_ids = {str(chunk.get("document_id")) for chunk in chunks if chunk.get("document_id")}
    if document_ids:
        return len(document_ids) == 1
    filenames = {str(chunk.get("filename")) for chunk in chunks if chunk.get("filename")}
    return len(filenames) == 1


def generic_fallback_document_chunks(
    chunks: list[dict[str, Any]],
    question: str,
) -> list[dict[str, Any]]:
    if index_has_single_document(chunks):
        return chunks

    requested_suffix = requested_document_suffix(question)
    if not requested_suffix:
        return []

    matching_chunks = [
        chunk
        for chunk in chunks
        if Path(str(chunk.get("filename", ""))).suffix.lower() == requested_suffix
    ]
    if index_has_single_document(matching_chunks):
        return matching_chunks
    return []


def requested_document_suffix(question: str) -> str | None:
    tokens = set(re.findall(r"[a-zA-Z0-9_]+", question.lower()))
    if "pdf" in tokens:
        return ".pdf"
    if "markdown" in tokens or "md" in tokens:
        return ".md"
    if "text" in tokens or "txt" in tokens:
        return ".txt"
    return None


def matches_document_filter(chunk: dict[str, Any], document_filter: str) -> bool:
    needle = document_filter.strip().lower()
    if not needle:
        return True
    values = [
        str(chunk.get("document_id", "")),
        str(chunk.get("filename", "")),
        str(chunk.get("source_path", "")),
    ]
    for value in values:
        lowered = value.lower()
        if lowered == needle:
            return True
        if value and Path(value).name.lower() == needle:
            return True
    return False


def summarize_index_documents(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        document_id = str(chunk.get("document_id") or chunk.get("filename") or "unknown")
        summary = documents.setdefault(
            document_id,
            {
                "document_id": document_id,
                "filename": chunk.get("filename", "unknown"),
                "source_path": chunk.get("source_path"),
                "source_type": chunk.get("source_type"),
                "chunk_count": 0,
                "page_numbers": set(),
            },
        )
        summary["chunk_count"] += 1
        page_number = chunk.get("page_number")
        if page_number is not None:
            summary["page_numbers"].add(page_number)

    items = []
    for summary in documents.values():
        page_numbers = summary.pop("page_numbers")
        summary["page_count"] = len(page_numbers)
        items.append(summary)
    return sorted(items, key=lambda item: (str(item["filename"]).lower(), str(item["document_id"])))


def build_faiss_index(matrix: np.ndarray) -> Any | None:
    faiss = import_optional_faiss()
    if faiss is None:
        return None
    index = faiss.IndexFlatIP(int(matrix.shape[1]))
    index.add(matrix.astype("float32"))
    return index


def import_optional_faiss() -> Any | None:
    if importlib.util.find_spec("faiss") is None:
        return None
    import faiss

    return faiss


def normalized_backend_name(backend: str) -> str:
    if backend == "dense":
        return "dense_lsa"
    return backend


def validate_backend(backend: str) -> None:
    if backend not in {"tfidf", HYBRID_BACKEND} | DENSE_BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}")


def load_index(index_dir: Path) -> RagIndex:
    path = index_dir.expanduser() / INDEX_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"RAG index not found: {path}")
    with path.open("rb") as file_obj:
        loaded = pickle.load(file_obj)
    if not isinstance(loaded, RagIndex):
        raise TypeError(f"Unexpected index object in {path}: {type(loaded)!r}")
    faiss_path = path.parent / FAISS_INDEX_FILENAME
    if faiss_path.exists():
        faiss = import_optional_faiss()
        if faiss is not None:
            loaded.faiss_index = faiss.read_index(str(faiss_path))
    return loaded
