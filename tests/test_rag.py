import json

import adip.rag.rerank as rerank_module
from adip.rag.chunks import read_chunks_jsonl
from adip.rag.evaluate import evaluate
from adip.rag.rerank import rerank_results
from adip.rag.retriever import build_index, load_index


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
