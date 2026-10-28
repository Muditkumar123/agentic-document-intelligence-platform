import importlib.util
import json
import math

import pytest

from adip.rag import onnx_reranker, rerank
from adip.rag.onnx_reranker import (
    FP32_FILENAME,
    INT8_FILENAME,
    METADATA_FILENAME,
    export_cross_encoder,
    score_onnx_pairs,
)
from adip.rag.rerank import ONNX_RERANKER_ENV, rerank_results, score_cross_encoder_pairs
from adip.rag.retriever import RetrievedChunk

HAVE_RUNTIMES = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "onnx", "onnxruntime"))
needs_runtimes = pytest.mark.skipif(not HAVE_RUNTIMES, reason="needs torch, transformers, onnx and onnxruntime")

QUESTION = "how long may personal data be kept"
TEXTS = [
    "personal data may be kept no longer than necessary",
    "tls removes static rsa key exchange",
    "a breach must be reported within hours",
    "data minimisation keeps only what is needed " * 5,
]


@pytest.fixture(scope="module")
def tiny_cross_encoder(tmp_path_factory):
    """A randomly initialised two-layer BERT cross-encoder with its own vocabulary: no download needed."""
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

    folder = tmp_path_factory.mktemp("tiny-cross-encoder")
    words = sorted({word for text in [QUESTION, *TEXTS] for word in text.split()})
    vocab = folder / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]), encoding="utf-8")
    tokenizer = BertTokenizer(str(vocab), model_max_length=128)
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        num_labels=1,
    )
    BertForSequenceClassification(config).eval().save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    exported = tmp_path_factory.mktemp("tiny-onnx")
    export_cross_encoder(str(folder), exported, local_files_only=True, quantize=True)
    return str(folder), exported


@needs_runtimes
def test_onnx_scores_match_pytorch(tiny_cross_encoder):
    model_dir, exported = tiny_cross_encoder

    reference = score_cross_encoder_pairs(QUESTION, TEXTS, model_name=model_dir, device="cpu", batch_size=3)
    onnx = score_onnx_pairs(QUESTION, TEXTS, str(exported / FP32_FILENAME), batch_size=3)

    assert len(onnx) == len(TEXTS)
    assert max(abs(a - b) for a, b in zip(reference, onnx)) < 1e-4


@needs_runtimes
def test_int8_model_scores_every_pair(tiny_cross_encoder):
    _, exported = tiny_cross_encoder

    scores = score_onnx_pairs(QUESTION, TEXTS, str(exported / INT8_FILENAME))

    assert (exported / INT8_FILENAME).stat().st_size < (exported / FP32_FILENAME).stat().st_size
    assert len(scores) == len(TEXTS) and all(math.isfinite(score) for score in scores)


@needs_runtimes
def test_rerank_results_gives_the_same_order_on_both_runtimes(tiny_cross_encoder, monkeypatch):
    model_dir, exported = tiny_cross_encoder
    monkeypatch.delenv(ONNX_RERANKER_ENV, raising=False)
    retrieved = [
        RetrievedChunk(chunk={"chunk_id": f"c{i}", "text": text}, score=0.1, rank=i + 1) for i, text in enumerate(TEXTS)
    ]

    torch_order = rerank_results(QUESTION, retrieved, reranker="cross_encoder", cross_encoder_model=model_dir, cross_encoder_device="cpu")
    onnx_order = rerank_results(QUESTION, retrieved, reranker="cross_encoder", cross_encoder_onnx=str(exported / FP32_FILENAME))

    assert [item.chunk["chunk_id"] for item in onnx_order] == [item.chunk["chunk_id"] for item in torch_order]


def test_onnx_scoring_validates_input_before_loading_anything():
    assert score_onnx_pairs(QUESTION, [], "missing.onnx") == []
    with pytest.raises(ValueError):
        score_onnx_pairs(QUESTION, TEXTS, "missing.onnx", batch_size=0)


def test_server_setting_routes_the_exported_model_to_onnx(tmp_path, monkeypatch):
    (tmp_path / METADATA_FILENAME).write_text(json.dumps({"model_name": "exported/model"}), encoding="utf-8")
    monkeypatch.setenv(ONNX_RERANKER_ENV, str(tmp_path / FP32_FILENAME))
    calls = []
    monkeypatch.setattr(onnx_reranker, "score_onnx_pairs", lambda q, texts, path, **_: calls.append(("onnx", path)) or [0.0, 1.0])
    monkeypatch.setattr(rerank, "score_cross_encoder_candidates", lambda **kw: calls.append(("torch", kw["model_name"])) or [0.0, 1.0])
    retrieved = [RetrievedChunk(chunk={"chunk_id": f"c{i}", "text": text}, score=0.1, rank=i + 1) for i, text in enumerate(TEXTS[:2])]

    onnx_order = rerank_results(QUESTION, retrieved, reranker="cross_encoder", cross_encoder_model="exported/model")
    rerank_results(QUESTION, retrieved, reranker="cross_encoder", cross_encoder_model="some/other-model")

    assert calls == [("onnx", str(tmp_path / FP32_FILENAME)), ("torch", "some/other-model")]
    assert [item.chunk["chunk_id"] for item in onnx_order] == ["c1", "c0"]


def test_server_setting_without_metadata_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setenv(ONNX_RERANKER_ENV, str(tmp_path / FP32_FILENAME))
    retrieved = [RetrievedChunk(chunk={"chunk_id": "c0", "text": TEXTS[0]}, score=0.1, rank=1)]

    with pytest.raises(FileNotFoundError):
        rerank_results(QUESTION, retrieved, reranker="cross_encoder")
