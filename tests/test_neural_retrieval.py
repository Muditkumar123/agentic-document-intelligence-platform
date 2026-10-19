"""Neural embeddings, the Qdrant vector store, and cross-encoder reranking (hermetic: fake models)."""

import hashlib
import re

import numpy as np
import pytest

import adip.rag.embeddings as embeddings
import adip.rag.rerank as rerank_module
from adip.agents.multihop_eval import paraphrase_retrieval
from adip.agents.runner import run_agent
from adip.rag.embeddings import resolve_embedding_model
from adip.rag.rerank import cross_encoder_max_length
from adip.rag.retriever import INDEX_FILENAME, build_index, load_index


class FakeEmbeddingModel:
    """Deterministic hashed bag-of-words vectors; records which encoder each call used."""

    def __init__(self):
        self.calls = []

    def _vectors(self, texts):
        matrix = np.zeros((len(texts), 64), dtype="float32")
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]{3,}", text.lower()):
                matrix[row, int(hashlib.md5(token.encode()).hexdigest(), 16) % 64] += 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1.0, norms)

    def encode_document(self, texts, **_):
        self.calls.append("document")
        return self._vectors(texts)

    def encode_query(self, texts, **_):
        self.calls.append("query")
        return self._vectors(texts)


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeEmbeddingModel()
    monkeypatch.setattr(embeddings, "get_embedding_model", lambda name, device=None: model)
    return model


def make_chunk(chunk_id, text, filename):
    return {
        "chunk_id": chunk_id,
        "document_id": f"doc_{filename}",
        "filename": filename,
        "source_path": f"/tmp/{filename}",
        "source_type": "md",
        "checksum": "abc",
        "page_number": 1,
        "chunk_index": 0,
        "text": text,
        "token_count": len(text.split()),
        "char_count": len(text),
        "metadata": {},
    }


CHUNKS = [
    make_chunk("oauth_tls", "The authorization server must require TLS for every token endpoint request.", "oauth.md"),
    make_chunk("oauth_refresh", "Refresh tokens are credentials used to obtain new access tokens.", "oauth.md"),
    make_chunk("dns_label", "Each DNS label must be 63 octets or less in length.", "dns.md"),
    make_chunk("tls_early", "TLS 1.3 clients may send early data in the first flight.", "tls.md"),
    make_chunk("gdpr_breach", "A personal data breach must be notified within 72 hours.", "gdpr.md"),
]
QUESTIONS = ["Which protocol must the token endpoint require?", "How long may a DNS label be?", "early data first flight"]


def test_named_embedding_models_resolve_to_safe_weights(monkeypatch):
    assert resolve_embedding_model("bge-m3").model_id == "BAAI/bge-m3"
    assert resolve_embedding_model("bge-m3").revision == "refs/pr/130"  # safetensors conversion
    assert resolve_embedding_model("BGE-M3").model_id == "BAAI/bge-m3"
    assert resolve_embedding_model("qwen3-embedding-0.6b").model_id == "Qwen/Qwen3-Embedding-0.6B"
    assert resolve_embedding_model("some-org/model").model_id == "some-org/model"
    monkeypatch.delenv(embeddings.ALLOW_DOWNLOAD_ENV, raising=False)
    assert not embeddings.downloads_allowed()
    monkeypatch.setenv(embeddings.ALLOW_DOWNLOAD_ENV, "1")
    assert embeddings.downloads_allowed()


def test_neural_dense_index_encodes_documents_and_queries_separately(fake_model):
    index = build_index(CHUNKS, backend="dense", embedding_model="bge-m3", use_faiss=False)
    results = index.search("How long may a DNS label be?", top_k=2)

    assert index.backend == "sentence_transformers"
    assert index.metadata["embedding_model_id"] == "BAAI/bge-m3"
    assert fake_model.calls == ["document", "query"]
    assert results[0].chunk["chunk_id"] == "dns_label"


@pytest.mark.parametrize("backend", ["dense", "hybrid"])
def test_qdrant_store_returns_the_same_results_as_in_memory_search(fake_model, backend):
    pytest.importorskip("qdrant_client")
    memory = build_index(CHUNKS, backend=backend, embedding_model="bge-m3", use_faiss=False)
    qdrant = build_index(CHUNKS, backend=backend, embedding_model="bge-m3", vector_store="qdrant")

    for question in QUESTIONS:
        # Chunks tied at score 0 (nothing in common with the query) have no defined order.
        expected = [(r.chunk["chunk_id"], round(r.score, 5)) for r in memory.search(question, top_k=3) if r.score > 0]
        actual = [(r.chunk["chunk_id"], round(r.score, 5)) for r in qdrant.search(question, top_k=3) if r.score > 0]
        assert actual == expected and actual
    filtered = qdrant.search("Which protocol must the token endpoint require?", top_k=3, document_filter="dns.md")
    assert {r.chunk["filename"] for r in filtered} == {"dns.md"}
    qdrant.vector_store_client.close()


def test_qdrant_index_round_trips_with_vectors_outside_the_archive(fake_model, tmp_path):
    pytest.importorskip("qdrant_client")
    index = build_index(CHUNKS, backend="hybrid", embedding_model="bge-m3", vector_store="qdrant")
    before = [[r.chunk["chunk_id"] for r in index.search(q, top_k=3)] for q in QUESTIONS]
    index.save(tmp_path / "index")
    index.vector_store_client.close()

    with np.load(tmp_path / "index" / INDEX_FILENAME, allow_pickle=False) as archive:
        stored = set(archive.files)
    loaded = load_index(tmp_path / "index")
    after = [[r.chunk["chunk_id"] for r in loaded.search(q, top_k=3)] for q in QUESTIONS]

    assert (tmp_path / "index" / "qdrant").is_dir()
    assert "matrix" not in stored  # the dense vectors live in Qdrant
    assert loaded.vector_store == "qdrant"
    assert loaded.vector_store_client.count() == len(CHUNKS)
    assert after == before
    loaded.vector_store_client.close()


def test_tfidf_backend_rejects_a_vector_store():
    with pytest.raises(ValueError, match="no dense vectors"):
        build_index(CHUNKS, backend="tfidf", vector_store="qdrant")


def test_cross_encoder_context_follows_the_model_capped_at_1024_tokens():
    class Tokenizer:
        def __init__(self, limit):
            self.model_max_length = limit

    assert cross_encoder_max_length(Tokenizer(512)) == 512
    assert cross_encoder_max_length(Tokenizer(8192)) == 1024
    assert cross_encoder_max_length(Tokenizer(int(1e30))) == 1024


def test_agent_reranks_each_search_stream_against_its_own_query(monkeypatch):
    scored_queries = []

    def fake_cross_encoder(question, retrieved, **_):
        scored_queries.append(question)
        terms = set(re.findall(r"[a-z0-9]{3,}", question.lower()))
        return [float(len(terms & set(re.findall(r"[a-z0-9]{3,}", item.chunk["text"].lower())))) for item in retrieved]

    monkeypatch.setattr(rerank_module, "score_cross_encoder_candidates", fake_cross_encoder)
    index = build_index(CHUNKS)
    question = "Which protocol must the token endpoint require, and how long may a DNS label be?"

    state = run_agent(
        question, index, task="qa", top_k=2, model_profile="extractive_baseline", reranker="cross_encoder"
    ).state

    assert scored_queries == [question, *state.sub_questions]
    assert {item["chunk"]["chunk_id"] for item in state.retrieved} >= {"oauth_tls", "dns_label"}
    assert state.metrics["reranker"] == "cross_encoder"


def test_paraphrase_retrieval_scores_hit_rate_and_mrr():
    index = build_index(CHUNKS)
    rows = [
        {"question": "How long may a DNS label be?", "expected_substrings": ["63 octets"]},
        {"question": "zzz unrelated", "expected_substrings": ["never present"]},
    ]

    result = paraphrase_retrieval(index, rows, top_k=3)

    assert result == {"question_count": 2.0, "hit_rate": 0.5, "mrr": 0.5}
