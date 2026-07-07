import json

import numpy as np
import pytest

import adip.rag.rerank as rerank_module
from adip.rag.bm25 import build_bm25_index
from adip.rag.chunks import read_chunks_jsonl
from adip.rag.evaluate import aggregate_by_category, evaluate, read_golden
from adip.rag.rerank import rerank_results, resolve_candidate_k
from adip.rag.retriever import (
    build_index,
    load_index,
    matches_document_filter,
    reciprocal_rank_fusion,
)


def make_chunk(chunk_id, text):
    return {
        "chunk_id": chunk_id,
        "document_id": "doc_test",
        "filename": "sample.md",
        "source_path": "/tmp/sample.md",
        "source_type": "md",
        "checksum": "abc123",
        "page_number": 1,
        "chunk_index": 0,
        "text": text,
        "token_count": len(text.split()),
        "char_count": len(text),
        "metadata": {},
    }


def make_document_chunk(document_id, chunk_id, filename, chunk_index, text):
    chunk = make_chunk(chunk_id, text)
    chunk["document_id"] = document_id
    chunk["filename"] = filename
    chunk["chunk_index"] = chunk_index
    chunk["page_number"] = chunk_index + 1
    return chunk


def test_build_save_load_and_search_index(tmp_path):
    chunks = [
        make_chunk("chunk_rag", "Retrieval augmented generation searches relevant document chunks."),
        make_chunk("chunk_ops", "MLflow and DVC track experiments and datasets."),
    ]

    index = build_index(chunks)
    index_path = tmp_path / "vector_index"
    index.save(index_path)

    loaded = load_index(index_path)
    results = loaded.search("How does retrieval find chunks?", top_k=1)

    assert results[0].chunk["chunk_id"] == "chunk_rag"
    assert results[0].score > 0
    assert "sample.md p.1" in results[0].citation_label


def test_single_document_generic_summary_falls_back_to_representative_chunks():
    index = build_index(
        [
            make_document_chunk(
                "doc_simon",
                "chunk_title",
                "simon.pdf",
                0,
                "Deep Learning Assisted Differential Cryptanalysis for the Lightweight Cipher SIMON.",
            ),
            make_document_chunk(
                "doc_simon",
                "chunk_method",
                "simon.pdf",
                1,
                "The authors train neural distinguishers for SIMON32/64.",
            ),
        ]
    )

    results = index.search("give me a summary of this pdf file", top_k=2)

    assert [item.chunk["chunk_id"] for item in results] == ["chunk_title", "chunk_method"]
    assert [item.score for item in results] == [0.0, 0.0]


def test_generic_pdf_summary_uses_only_indexed_pdf_when_other_files_exist():
    index = build_index(
        [
            make_document_chunk(
                "doc_simon",
                "chunk_pdf_title",
                "simon.pdf",
                0,
                "Deep Learning Assisted Differential Cryptanalysis for the Lightweight Cipher SIMON.",
            ),
            make_document_chunk(
                "doc_simon",
                "chunk_pdf_method",
                "simon.pdf",
                1,
                "The authors train neural distinguishers for SIMON32/64.",
            ),
            make_document_chunk(
                "doc_notes",
                "chunk_notes",
                "notes.md",
                0,
                "Operational notes for a document intelligence platform.",
            ),
        ]
    )

    results = index.search("give me a summary of this pdf file", top_k=2)

    assert [item.chunk["chunk_id"] for item in results] == ["chunk_pdf_title", "chunk_pdf_method"]


def test_document_filter_matches_partial_filename():
    chunk = {
        "document_id": "doc_abc123",
        "filename": "GenNet.pdf",
        "source_path": "data/raw/GenNet.pdf",
    }
    assert matches_document_filter(chunk, "GenNet.pdf")
    assert matches_document_filter(chunk, "gennet")
    assert matches_document_filter(chunk, "doc_abc123")
    assert not matches_document_filter(chunk, "transformers")


def test_document_filter_limits_retrieval_to_selected_document():
    index = build_index(
        [
            make_document_chunk(
                "doc_simon",
                "chunk_simon",
                "simon.pdf",
                0,
                "SIMON neural distinguishers use deep learning for differential cryptanalysis.",
            ),
            make_document_chunk(
                "doc_notes",
                "chunk_notes",
                "notes.md",
                0,
                "SIMON appears here only as an unrelated note.",
            ),
        ]
    )

    results = index.search("What does SIMON use?", top_k=2, document_filter="doc_notes")

    assert len(results) == 1
    assert results[0].chunk["chunk_id"] == "chunk_notes"


def test_document_filter_supports_generic_summary_for_selected_pdf():
    index = build_index(
        [
            make_document_chunk(
                "doc_one",
                "chunk_one",
                "one.pdf",
                0,
                "First PDF has no query overlap.",
            ),
            make_document_chunk(
                "doc_two",
                "chunk_two",
                "two.pdf",
                0,
                "Second PDF has no query overlap.",
            ),
        ]
    )

    results = index.search("summarize this pdf", top_k=1, document_filter="two.pdf")

    assert len(results) == 1
    assert results[0].chunk["chunk_id"] == "chunk_two"


def test_generic_summary_does_not_guess_across_multiple_documents():
    index = build_index(
        [
            make_document_chunk("doc_one", "chunk_one", "one.pdf", 0, "Alpha content."),
            make_document_chunk("doc_two", "chunk_two", "two.pdf", 0, "Beta content."),
        ]
    )

    assert index.search("give me a summary of this pdf file", top_k=2) == []


def test_build_save_load_and_search_dense_lsa_index(tmp_path):
    chunks = [
        make_chunk("chunk_rag", "Semantic retrieval finds relevant document evidence."),
        make_chunk("chunk_ops", "Deployment monitoring tracks latency and failures."),
    ]

    index = build_index(
        chunks,
        backend="dense",
        embedding_model="lsa",
        dense_dimensions=8,
        use_faiss=False,
    )
    index_path = tmp_path / "dense_index"
    index.save(index_path)

    loaded = load_index(index_path)
    results = loaded.search("How do we retrieve relevant evidence?", top_k=2)

    assert loaded.backend == "dense_lsa"
    assert loaded.metadata["embedding_model"] == "lsa"
    assert len(results) == 2
    assert {item.chunk["chunk_id"] for item in results} <= {"chunk_rag", "chunk_ops"}


def test_lexical_reranker_can_promote_candidate():
    chunks = [
        make_chunk("chunk_generic", "General platform monitoring and reports."),
        make_chunk(
            "chunk_specific",
            "AgentOps trace events store node name, input summary, output summary, timing, and errors.",
        ),
    ]
    index = build_index(chunks)
    candidates = index.search("What does each AgentOps trace event store?", top_k=2)
    reversed_candidates = list(reversed(candidates))

    reranked = rerank_results(
        "What does each AgentOps trace event store?",
        reversed_candidates,
        reranker="lexical",
        top_k=2,
        original_score_weight=0.0,
    )

    assert reranked[0].chunk["chunk_id"] == "chunk_specific"
    assert reranked[0].rank == 1


def test_cross_encoder_reranker_can_promote_candidate(monkeypatch):
    chunks = [
        make_chunk("chunk_generic", "General platform monitoring and reports."),
        make_chunk(
            "chunk_specific",
            "AgentOps trace events store node name, input summary, output summary, timing, and errors.",
        ),
    ]
    candidates = [
        rerank_module.RetrievedChunk(chunk=chunks[0], score=1.0, rank=1),
        rerank_module.RetrievedChunk(chunk=chunks[1], score=0.5, rank=2),
    ]

    def fake_cross_encoder_pairs(**kwargs):
        return [0.1, 2.0]

    monkeypatch.setattr(rerank_module, "score_cross_encoder_pairs", fake_cross_encoder_pairs)

    reranked = rerank_results(
        "What does each AgentOps trace event store?",
        candidates,
        reranker="cross_encoder",
        top_k=2,
        original_score_weight=0.0,
    )

    assert reranked[0].chunk["chunk_id"] == "chunk_specific"
    assert reranked[0].rank == 1


def test_read_chunks_jsonl_validates_required_fields(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(json.dumps(make_chunk("chunk_one", "hello world")) + "\n", encoding="utf-8")

    chunks = read_chunks_jsonl(chunks_path)

    assert chunks[0]["chunk_id"] == "chunk_one"


def test_evaluate_reports_hit_rate(tmp_path):
    chunks = [
        make_chunk("chunk_ingest", "The platform can ingest long documents and split text into overlapping chunks."),
        make_chunk("chunk_other", "A separate note about deployment monitoring."),
    ]
    index_path = tmp_path / "vector_index"
    build_index(chunks).save(index_path)

    golden_path = tmp_path / "golden.jsonl"
    golden_path.write_text(
        json.dumps(
            {
                "question": "How can the platform ingest long documents?",
                "expected_substrings": ["split text into overlapping chunks"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluate(index_path, golden_path, top_k=2)

    assert report["question_count"] == 1
    assert report["hit_rate_at_k"] == 1.0
    assert report["mrr"] == 1.0


def test_aggregate_by_category_slices_hit_rate_and_mrr():
    results = [
        {"category": "legal", "hit": True, "rank": 1},
        {"category": "legal", "hit": False, "rank": None},
        {"category": "finance", "hit": True, "rank": 2},
        {"category": None, "hit": True, "rank": 1},
    ]
    hit_rate, mrr = aggregate_by_category(results)
    assert hit_rate["legal"] == 0.5
    assert mrr["legal"] == 0.5  # one hit at rank 1, one miss, averaged over 2
    assert hit_rate["finance"] == 1.0
    assert mrr["finance"] == 0.5  # single hit at rank 2 -> 1/2
    assert hit_rate["uncategorized"] == 1.0  # None category bucketed


def test_read_golden_preserves_category_field(tmp_path):
    golden_path = tmp_path / "golden.jsonl"
    golden_path.write_text(
        json.dumps({"question": "q?", "expected_substrings": ["a"], "category": "legal"}) + "\n",
        encoding="utf-8",
    )
    rows = read_golden(golden_path)
    assert rows[0]["category"] == "legal"


def test_evaluate_emits_by_category_slices(tmp_path):
    chunks = [
        make_chunk("chunk_law", "Personal data shall be processed lawfully under the regulation."),
        make_chunk("chunk_fin", "A mutual fund pools money from many investors into a portfolio."),
    ]
    index_path = tmp_path / "vector_index"
    build_index(chunks).save(index_path)

    golden_path = tmp_path / "golden.jsonl"
    golden_path.write_text(
        json.dumps({"question": "How is personal data processed?", "expected_substrings": ["processed lawfully"], "category": "legal"})
        + "\n"
        + json.dumps({"question": "What is a mutual fund?", "expected_substrings": ["pools money from many investors"], "category": "finance"})
        + "\n",
        encoding="utf-8",
    )

    report = evaluate(index_path, golden_path, top_k=2)
    assert set(report["hit_rate_by_category"]) == {"legal", "finance"}
    assert report["hit_rate_by_category"]["legal"] == 1.0
    assert report["mrr_by_category"]["finance"] == 1.0
    assert report["results"][0]["category"] == "legal"


def test_evaluate_with_reranker_reports_reranker_metadata(tmp_path):
    chunks = [
        make_chunk("chunk_ingest", "The platform can ingest long documents and split text into overlapping chunks."),
        make_chunk("chunk_other", "A separate note about deployment monitoring."),
    ]
    index_path = tmp_path / "vector_index"
    build_index(chunks).save(index_path)

    golden_path = tmp_path / "golden.jsonl"
    golden_path.write_text(
        json.dumps(
            {
                "question": "How can the platform ingest long documents?",
                "expected_substrings": ["split text into overlapping chunks"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluate(
        index_path,
        golden_path,
        top_k=1,
        candidate_k=2,
        reranker="lexical",
        rerank_weight=0.1,
    )

    assert report["reranker"] == "lexical"
    assert report["candidate_k"] == 2
    assert report["hit_rate_at_k"] == 1.0


def test_bm25_prefers_document_with_rare_query_term():
    texts = [
        "the platform stores documents and answers questions about documents",
        "quarterly financial statements disclose revenue and liabilities",
        "the platform stores documents",
    ]
    bm25 = build_bm25_index(texts)
    scores = bm25.scores("what were the revenue liabilities?")

    assert scores[1] > scores[0]
    assert scores[1] > scores[2]
    assert scores[0] == 0.0


def test_bm25_all_stopword_query_scores_zero():
    bm25 = build_bm25_index(["alpha beta gamma", "delta epsilon"])
    assert bm25.scores("the of and").sum() == 0.0


def test_reciprocal_rank_fusion_top_of_both_lists_scores_one():
    fused = reciprocal_rank_fusion(
        score_lists=[np.array([5.0, 1.0]), np.array([7.0, 2.0])],
        weights=[0.5, 0.5],
        candidate_indices=[0, 1],
    )
    assert fused[0] == pytest.approx(1.0)
    assert fused[1] < fused[0]


def test_reciprocal_rank_fusion_ignores_nonpositive_component_scores():
    fused = reciprocal_rank_fusion(
        score_lists=[np.array([0.0, 2.0]), np.array([0.0, 1.0])],
        weights=[0.5, 0.5],
        candidate_indices=[0, 1],
    )
    assert fused[0] == 0.0
    assert fused[1] == pytest.approx(1.0)


def test_reciprocal_rank_fusion_respects_component_weights():
    scores_a = np.array([3.0, 2.0, 1.0])
    scores_b = np.array([1.0, 3.0, 2.0])
    lexical_only = reciprocal_rank_fusion(
        score_lists=[scores_a, scores_b],
        weights=[1.0, 0.0],
        candidate_indices=[0, 1, 2],
    )
    dense_only = reciprocal_rank_fusion(
        score_lists=[scores_a, scores_b],
        weights=[0.0, 1.0],
        candidate_indices=[0, 1, 2],
    )
    assert max(lexical_only, key=lexical_only.get) == 0
    assert max(dense_only, key=dense_only.get) == 1


def test_build_save_load_and_search_hybrid_index(tmp_path):
    chunks = [
        make_chunk("chunk_rag", "Retrieval augmented generation searches relevant document chunks."),
        make_chunk("chunk_ops", "MLflow and DVC track experiments and datasets."),
        make_chunk("chunk_deploy", "Docker images ship the API to production servers."),
    ]

    index = build_index(chunks, backend="hybrid")
    index_path = tmp_path / "hybrid_index"
    index.save(index_path)

    loaded = load_index(index_path)
    results = loaded.search("How does retrieval find document chunks?", top_k=2)

    assert loaded.backend == "hybrid"
    assert loaded.metadata["fusion"] == "weighted_rrf"
    assert results[0].chunk["chunk_id"] == "chunk_rag"
    assert 0 < results[0].score <= 1.0


def test_hybrid_index_respects_document_filter():
    chunks = [
        make_document_chunk("doc_a", "chunk_a", "alpha.md", 0, "Alpha discusses retrieval quality."),
        make_document_chunk("doc_b", "chunk_b", "beta.md", 0, "Beta discusses retrieval quality."),
    ]
    index = build_index(chunks, backend="hybrid")

    results = index.search("retrieval quality", top_k=2, document_filter="beta.md")

    assert [item.chunk["chunk_id"] for item in results] == ["chunk_b"]


def test_hybrid_index_validates_dense_weight():
    with pytest.raises(ValueError):
        build_index([make_chunk("chunk_a", "some text")], backend="hybrid", hybrid_dense_weight=1.5)


def test_resolve_candidate_k_widens_pool_only_when_reranking():
    assert resolve_candidate_k(3, None, "none") == 3
    assert resolve_candidate_k(3, None, "lexical") == 10
    assert resolve_candidate_k(5, None, "cross_encoder") == 15
    assert resolve_candidate_k(3, 7, "lexical") == 7
    with pytest.raises(ValueError):
        resolve_candidate_k(5, 3, "lexical")


def test_expand_query_keywords_adds_content_and_morphological_variants():
    from adip.rag.rewrite import expand_query_keywords

    variants = expand_query_keywords("What are the principles for processing personal data?")

    assert variants[0] == "What are the principles for processing personal data?"
    assert any("principles processing personal data" in v for v in variants)
    assert any("principle" in v and "principles" not in v for v in variants)
    assert len(variants) == len({v.lower() for v in variants})


def test_rewrite_question_modes():
    from adip.rag.rewrite import rewrite_question

    assert rewrite_question("What is TLS?", rewriter="none") == ["What is TLS?"]
    keyword_variants = rewrite_question("What are safe HTTP methods?", rewriter="keywords")
    assert keyword_variants[0] == "What are safe HTTP methods?"
    assert len(keyword_variants) >= 2

    def fake(question):
        return ["Which web verbs are safe?", "  ", "What are safe HTTP methods?"]

    llm_variants = rewrite_question("What are safe HTTP methods?", rewriter="llm", llm_rewriter=fake)
    assert llm_variants == ["What are safe HTTP methods?", "Which web verbs are safe?"]

    with pytest.raises(ValueError):
        rewrite_question("q", rewriter="mystery")
    with pytest.raises(ValueError):
        rewrite_question("q", rewriter="llm", llm_rewriter=None)


def test_fuse_ranked_lists_promotes_consensus_chunks():
    from adip.rag.retriever import RetrievedChunk
    from adip.rag.rewrite import fuse_ranked_lists

    chunk_a = {"chunk_id": "a", "text": "alpha"}
    chunk_b = {"chunk_id": "b", "text": "beta"}
    chunk_c = {"chunk_id": "c", "text": "gamma"}
    list_one = [RetrievedChunk(chunk_a, 0.9, 1), RetrievedChunk(chunk_b, 0.5, 2)]
    list_two = [RetrievedChunk(chunk_c, 0.8, 1), RetrievedChunk(chunk_a, 0.4, 2)]

    fused = fuse_ranked_lists([list_one, list_two])

    assert fused[0].chunk["chunk_id"] == "a"  # appears in both lists
    assert fused[0].rank == 1
    assert fused[0].score <= 1.0
    assert {item.chunk["chunk_id"] for item in fused} == {"a", "b", "c"}


def test_fuse_ranked_lists_rank_one_everywhere_scores_one():
    from adip.rag.retriever import RetrievedChunk
    from adip.rag.rewrite import fuse_ranked_lists

    chunk = {"chunk_id": "a", "text": "alpha"}
    fused = fuse_ranked_lists([[RetrievedChunk(chunk, 0.9, 1)], [RetrievedChunk(chunk, 0.7, 1)]])
    assert fused[0].score == pytest.approx(1.0)


def test_retrieve_with_rewrites_recovers_morphological_mismatch():
    from adip.rag.rewrite import retrieve_with_rewrites, rewrite_question

    chunks = [
        make_chunk("chunk_principle", "The principle of storage limitation restricts retention."),
        make_chunk("chunk_other", "Unrelated deployment notes about container images."),
    ]
    index = build_index(chunks)
    question = "What are the principles that restrict retention?"

    plain = index.search(question, top_k=1)
    variants = rewrite_question(question, rewriter="keywords")
    rewritten = retrieve_with_rewrites(index, variants, top_k=1)

    assert rewritten[0].chunk["chunk_id"] == "chunk_principle"
    assert plain == [] or plain[0].chunk["chunk_id"] == "chunk_principle"


def test_parse_rewrites_tolerates_numbering_and_prose():
    from adip.rag.rewrite import parse_rewrites

    text = """Here are the rewrites:
1. Which web verbs are considered safe?
2) What request methods cannot modify state?
- How do idempotent HTTP methods behave?
Short one
"""
    rewrites = parse_rewrites(text)
    assert rewrites == [
        "Which web verbs are considered safe?",
        "What request methods cannot modify state?",
        "How do idempotent HTTP methods behave?",
    ]


def test_evaluate_with_rewriter_reports_rewriter_metadata(tmp_path):
    chunks = [
        make_chunk("chunk_ingest", "The platform can ingest long documents and split text into overlapping chunks."),
    ]
    index_path = tmp_path / "vector_index"
    build_index(chunks).save(index_path)
    golden_path = tmp_path / "golden.jsonl"
    golden_path.write_text(
        json.dumps(
            {
                "question": "How does the platform ingest documents?",
                "expected_substrings": ["split text into overlapping chunks"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluate(index_path, golden_path, top_k=1, rewriter="keywords")

    assert report["rewriter"] == "keywords"
    assert report["hit_rate_at_k"] == 1.0
