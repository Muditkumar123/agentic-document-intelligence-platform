"""Request schemas for the HTTP API."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from adip.rag.rerank import DEFAULT_CROSS_ENCODER_MODEL


class RagQueryRequest(BaseModel):
    question: str = Field(min_length=1)
    index_path: Path = Path("data/processed/vector_index")
    document_filter: str | None = None
    top_k: int = Field(default=3, ge=1)
    candidate_k: int | None = Field(default=None, ge=1)
    reranker: Literal["none", "lexical", "cross_encoder"] = "none"
    rerank_weight: float = Field(default=0.25, ge=0)
    cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL
    cross_encoder_device: str | None = None
    cross_encoder_batch_size: int = Field(default=16, ge=1)
    allow_reranker_download: bool = False


class AgentRunRequest(BaseModel):
    question: str = Field(min_length=1)
    index_path: Path = Path("data/processed/vector_index")
    document_filter: str | None = None
    task: Literal["auto", "qa", "brief"] = "auto"
    domain: str = "general"
    top_k: int = Field(default=3, ge=1)
    llm_provider: Literal["extractive", "huggingface", "openai_compatible"] | None = None
    model_profile: str | None = "extractive_baseline"
    model_name: str | None = None
    endpoint_url: str | None = None
    api_key: str | None = None
    device: str = "cuda:1"
    max_new_tokens: int = Field(default=4096, ge=1)
    reasoning_effort: Literal["auto", "none", "low", "medium", "high"] = "auto"
    reasoning_provider: Literal["extractive", "huggingface", "openai_compatible"] | None = None
    reasoning_model_profile: str | None = None
    reasoning_model_name: str | None = None
    reasoning_endpoint_url: str | None = None
    reasoning_api_key: str | None = None
    reasoning_device: str | None = "cuda:1"
    reasoning_max_new_tokens: int = Field(default=256, ge=1)
    use_reasoning_planner: bool = False
    trace_dir: Path | None = Path("data/monitoring/agent_traces")


class RebuildIndexRequest(BaseModel):
    input_path: Path = Path("data/raw")
    chunks_path: Path = Path("data/processed/chunks.jsonl")
    index_path: Path = Path("data/processed/vector_index")
    chunk_size: int = Field(default=800, ge=1)
    chunk_overlap: int = Field(default=120, ge=0)
    backend: Literal["tfidf", "dense", "dense_lsa", "sentence_transformers"] = "tfidf"
    ngram_max: int = Field(default=2, ge=1)
    embedding_model: str = "lsa"
    dense_dimensions: int = Field(default=128, ge=1)
    use_faiss: bool = True


class ModelCheckRequest(BaseModel):
    model_name: str = Field(min_length=1)
    endpoint_url: str = Field(min_length=1)
    api_key: str = Field(min_length=1)
    max_new_tokens: int = Field(default=128, ge=1, le=512)
