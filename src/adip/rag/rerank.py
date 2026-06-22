"""Second-stage reranking helpers for retrieved chunks."""

from __future__ import annotations

import importlib.util
import math
import re
from collections import Counter
from functools import lru_cache
from typing import Iterable

from adip.rag.retriever import RetrievedChunk

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")
SUPPORTED_RERANKERS = {"none", "lexical", "cross_encoder"}
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank_results(
    question: str,
    retrieved: list[RetrievedChunk],
    reranker: str = "none",
    top_k: int | None = None,
    original_score_weight: float = 0.25,
    cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
    cross_encoder_device: str | None = None,
    cross_encoder_local_files_only: bool = True,
    cross_encoder_batch_size: int = 16,
) -> list[RetrievedChunk]:
    if reranker == "none":
        return retrieved[:top_k] if top_k is not None else retrieved
    if reranker not in SUPPORTED_RERANKERS:
        raise ValueError(f"Unsupported reranker: {reranker}")
    if original_score_weight < 0:
        raise ValueError("original_score_weight must be greater than or equal to 0")

    if reranker == "lexical":
        scores = score_lexical_candidates(question, retrieved)
    else:
        scores = score_cross_encoder_candidates(
            question=question,
            retrieved=retrieved,
            model_name=cross_encoder_model,
            device=cross_encoder_device,
            local_files_only=cross_encoder_local_files_only,
            batch_size=cross_encoder_batch_size,
        )

    scored = [
        RetrievedChunk(
            chunk=item.chunk,
            score=float(score) + (original_score_weight * item.score),
            rank=item.rank,
        )
        for item, score in zip(retrieved, scores)
    ]

    scored.sort(key=lambda item: item.score, reverse=True)
    limit = len(scored) if top_k is None else top_k
    return [
        RetrievedChunk(chunk=item.chunk, score=item.score, rank=rank)
        for rank, item in enumerate(scored[:limit], start=1)
    ]


def score_lexical_candidates(question: str, retrieved: list[RetrievedChunk]) -> list[float]:
    query_terms = tokenize(question)
    return [score_lexical_overlap(query_terms, item.chunk["text"]) for item in retrieved]


def score_lexical_overlap(query_terms: list[str], text: str) -> float:
    if not query_terms:
        return 0.0
    query_counts = Counter(query_terms)
    text_terms = tokenize(text)
    text_counts = Counter(text_terms)

    overlap = 0.0
    for term, query_count in query_counts.items():
        if term in text_counts:
            overlap += min(query_count, text_counts[term]) / math.sqrt(text_counts[term])

    coverage = len(set(query_terms) & set(text_terms)) / len(set(query_terms))
    phrase_bonus = contiguous_bigram_bonus(query_terms, text_terms)
    length_penalty = 1.0 / math.sqrt(max(1, len(text_terms) / 80))
    return ((overlap * 0.7) + (coverage * 2.0) + phrase_bonus) * length_penalty


def contiguous_bigram_bonus(query_terms: list[str], text_terms: list[str]) -> float:
    query_bigrams = set(pairwise(query_terms))
    text_bigrams = set(pairwise(text_terms))
    if not query_bigrams:
        return 0.0
    return len(query_bigrams & text_bigrams) / len(query_bigrams)


def pairwise(values: Iterable[str]) -> Iterable[tuple[str, str]]:
    values = list(values)
    return zip(values, values[1:])


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def score_cross_encoder_candidates(
    question: str,
    retrieved: list[RetrievedChunk],
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    device: str | None = None,
    local_files_only: bool = True,
    batch_size: int = 16,
) -> list[float]:
    texts = [item.chunk["text"] for item in retrieved]
    return score_cross_encoder_pairs(
        question=question,
        texts=texts,
        model_name=model_name,
        device=device,
        local_files_only=local_files_only,
        batch_size=batch_size,
    )


def score_cross_encoder_pairs(
    question: str,
    texts: list[str],
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    device: str | None = None,
    local_files_only: bool = True,
    batch_size: int = 16,
) -> list[float]:
    if not texts:
        return []
    if batch_size <= 0:
        raise ValueError("cross_encoder_batch_size must be greater than 0")

    bundle = load_cross_encoder_bundle(
        model_name=model_name,
        device=device or "",
        local_files_only=local_files_only,
    )
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    resolved_device = bundle["device"]

    import torch

    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            inputs = tokenizer(
                [question] * len(batch_texts),
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(resolved_device)
            logits = model(**inputs).logits
            if logits.ndim == 2 and logits.shape[-1] > 1:
                batch_scores = logits[:, -1]
            else:
                batch_scores = logits.reshape(-1)
            scores.extend(float(score) for score in batch_scores.detach().cpu().tolist())
    return scores


@lru_cache(maxsize=4)
def load_cross_encoder_bundle(
    model_name: str,
    device: str,
    local_files_only: bool,
) -> dict[str, object]:
    if importlib.util.find_spec("transformers") is None:
        raise ImportError("transformers is required for the cross_encoder reranker")
    if importlib.util.find_spec("torch") is None:
        raise ImportError("torch is required for the cross_encoder reranker")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model.to(resolved_device)
    return {
        "tokenizer": tokenizer,
        "model": model,
        "device": resolved_device,
    }


def validate_reranker(reranker: str) -> None:
    if reranker not in SUPPORTED_RERANKERS:
        raise ValueError(f"Unsupported reranker: {reranker}")
