"""Request schemas for the HTTP API.

Requests never carry filesystem paths: the server owns its data layout (see
``adip.api.settings``) and clients choose an index by backend name. Unknown
fields are rejected, so an old client still sending ``index_path`` gets a clear
422 instead of having the field silently ignored.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from adip.rag.rerank import DEFAULT_CROSS_ENCODER_MODEL

IndexBackend = Literal["tfidf", "dense", "hybrid"]
# A fixed choice, not any Hugging Face id or local path: clients don't pick what the server loads.
CrossEncoderModel = Literal["cross-encoder/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-v2-m3"]


class RagQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    backend: IndexBackend = "tfidf"
    document_filter: str | None = None
    top_k: int = Field(default=3, ge=1)
    candidate_k: int | None = Field(default=None, ge=1)
    reranker: Literal["none", "lexical", "cross_encoder"] = "none"
    rewriter: Literal["none", "keywords"] = "none"
    rerank_weight: float = Field(default=0.25, ge=0)
    cross_encoder_model: CrossEncoderModel = DEFAULT_CROSS_ENCODER_MODEL
    cross_encoder_device: str | None = None
    cross_encoder_batch_size: int = Field(default=16, ge=1)
    use_cache: bool = True


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    backend: IndexBackend = "tfidf"
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
    engine: Literal["auto", "langgraph", "sequential"] = "auto"
    # Retrieval loop: split multi-part questions, and let the verifier search again
    # (up to this many extra rounds) when the evidence does not cover the question.
    decomposer: Literal["none", "rules", "llm"] = "rules"
    max_retrieval_retries: int = Field(default=2, ge=0, le=4)
    # Rerank every search stream against its own query (see adip.agents.nodes.search_stream).
    reranker: Literal["none", "lexical", "cross_encoder"] = "none"
    cross_encoder_model: CrossEncoderModel = DEFAULT_CROSS_ENCODER_MODEL


class RebuildIndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_size: int = Field(default=800, ge=1)
    chunk_overlap: int = Field(default=120, ge=0)
    backend: Literal["tfidf", "dense", "dense_lsa", "sentence_transformers", "hybrid"] = "tfidf"
    ngram_max: int = Field(default=2, ge=1)
    embedding_model: str = "lsa"
    dense_dimensions: int = Field(default=128, ge=1)
    use_faiss: bool = True
    rrf_k: int = Field(default=60, ge=1)
    hybrid_dense_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    # Also rebuild the other standard-backend indexes (tfidf/dense/hybrid)
    # from the same chunks so switching backends never hits a stale index.
    all_backends: bool = False


class ModelCheckRequest(BaseModel):
    model_name: str = Field(min_length=1)
    endpoint_url: str = Field(min_length=1)
    api_key: str = Field(min_length=1)
    max_new_tokens: int = Field(default=128, ge=1, le=512)
