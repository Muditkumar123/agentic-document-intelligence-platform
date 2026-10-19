"""Neural text embeddings for dense and hybrid retrieval.

Wraps sentence-transformers with what retrieval actually needs:

- **Named models** with the settings they require. ``bge-m3`` (BAAI, 568M params,
  1024-dim) loads its safetensors weights from Hugging Face's official conversion
  (``refs/pr/130``), because the main branch only ships a pickle-based
  ``pytorch_model.bin`` that current transformers refuses to load on torch < 2.6.
  ``qwen3-embedding-0.6b`` (Qwen, 1024-dim) prefixes queries with its retrieval
  instruction. Any other name is passed to sentence-transformers as a Hugging Face id.
- **Query vs document encoding**, through ``encode_query`` / ``encode_document`` so
  models with asymmetric prompts get them.
- **One load per process**: the model is cached, so a query costs one forward pass,
  not a model load.
- **No surprise downloads**: models load from the local cache unless
  ``ADIP_ALLOW_MODEL_DOWNLOAD=1``. A 2 GB download should be a decision, not a
  side effect of the first query.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

ALLOW_DOWNLOAD_ENV = "ADIP_ALLOW_MODEL_DOWNLOAD"
DEVICE_ENV = "ADIP_EMBEDDING_DEVICE"
DEFAULT_MAX_SEQ_LENGTH = 1024
DEFAULT_BATCH_SIZE = 16


@dataclass(frozen=True)
class EmbeddingModelSpec:
    model_id: str
    revision: str | None = None
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH


KNOWN_EMBEDDING_MODELS: dict[str, EmbeddingModelSpec] = {
    "bge-m3": EmbeddingModelSpec("BAAI/bge-m3", revision="refs/pr/130"),
    "qwen3-embedding-0.6b": EmbeddingModelSpec("Qwen/Qwen3-Embedding-0.6B"),
}


def resolve_embedding_model(name: str) -> EmbeddingModelSpec:
    return KNOWN_EMBEDDING_MODELS.get(name.lower(), EmbeddingModelSpec(name))


def default_device() -> str:
    configured = os.environ.get(DEVICE_ENV)
    if configured:
        return configured
    if importlib.util.find_spec("torch") is not None:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    return "cpu"


def downloads_allowed() -> bool:
    return os.environ.get(ALLOW_DOWNLOAD_ENV) == "1"


@lru_cache(maxsize=4)
def load_sentence_transformer(
    model_id: str,
    revision: str | None,
    device: str,
    local_files_only: bool,
    max_seq_length: int,
) -> Any:
    if importlib.util.find_spec("sentence_transformers") is None:
        raise ImportError(
            "sentence-transformers is required for neural embedding models: "
            'pip install -e ".[retrieval]"'
        )
    from sentence_transformers import SentenceTransformer

    model_kwargs: dict[str, Any] = {}
    if device.startswith("cuda"):
        import torch

        model_kwargs["dtype"] = torch.float16
    model = SentenceTransformer(
        model_id,
        revision=revision,
        device=device,
        local_files_only=local_files_only,
        model_kwargs=model_kwargs or None,
    )
    model.max_seq_length = max_seq_length
    return model


def get_embedding_model(name: str, device: str | None = None) -> Any:
    spec = resolve_embedding_model(name)
    return load_sentence_transformer(
        spec.model_id,
        spec.revision,
        device or default_device(),
        not downloads_allowed(),
        spec.max_seq_length,
    )


def _encode(model: Any, texts: list[str], kind: str, batch_size: int) -> np.ndarray:
    encode = getattr(model, f"encode_{kind}", None) or model.encode
    embeddings = encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype="float32")


def encode_documents(texts: list[str], name: str, batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
    """L2-normalized float32 document embeddings, one row per text."""
    return _encode(get_embedding_model(name), texts, "document", batch_size)


def encode_queries(texts: list[str], name: str, batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
    """L2-normalized float32 query embeddings (with the model's query prompt, if any)."""
    return _encode(get_embedding_model(name), texts, "query", batch_size)


def embedding_model_metadata(name: str) -> dict[str, Any]:
    spec = resolve_embedding_model(name)
    return {"embedding_model_id": spec.model_id, "embedding_model_revision": spec.revision}
