"""Pickle-free persistence helpers for RAG indexes.

Indexes used to be written with ``pickle``, and ``pickle.load`` executes whatever
the file tells it to — loading an index from an untrusted location was a
code-execution risk. The index is now stored as plain data: one ``.npz`` archive
holding numeric arrays plus a UTF-8 JSON manifest, read back with
``np.load(allow_pickle=False)``. Fitted scikit-learn vectorizers are reduced to
their constructor parameters, vocabulary, and IDF weights, which is everything
``transform`` needs, and rebuilt on load.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Vectorizer constructor parameters that are plain JSON values. Anything else
# (custom tokenizers, preprocessors, callables) cannot be stored safely.
JSON_VECTORIZER_PARAMS = (
    "analyzer",
    "binary",
    "decode_error",
    "encoding",
    "input",
    "lowercase",
    "max_df",
    "max_features",
    "min_df",
    "ngram_range",
    "norm",
    "smooth_idf",
    "stop_words",
    "strip_accents",
    "sublinear_tf",
    "token_pattern",
    "use_idf",
)


class IndexFormatError(ValueError):
    """The index file is corrupt or was written by an unsupported version."""


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Index field of type {type(value).__name__} is not JSON-serializable")


def encode_json(payload: Any) -> np.ndarray:
    """JSON as a uint8 array, so it can live inside a pickle-free ``.npz`` archive."""
    text = json.dumps(payload, sort_keys=True, default=_json_default)
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8)


def decode_json(array: np.ndarray) -> Any:
    return json.loads(array.tobytes().decode("utf-8"))


def vectorizer_state(vectorizer: CountVectorizer) -> dict[str, Any]:
    """JSON-safe description of a fitted Count/TF-IDF vectorizer (without IDF weights)."""
    params = vectorizer.get_params()
    for name in ("preprocessor", "tokenizer", "vocabulary"):
        if params.get(name) is not None:
            raise ValueError(f"Cannot persist a vectorizer with a custom `{name}`")
    if params.get("analyzer") not in {"word", "char", "char_wb"}:
        raise ValueError("Cannot persist a vectorizer with a callable analyzer")
    stored = {name: params[name] for name in JSON_VECTORIZER_PARAMS if name in params}
    if isinstance(stored.get("stop_words"), (list, tuple, set, frozenset)):
        stored["stop_words"] = sorted(stored["stop_words"])
    if "ngram_range" in stored:
        stored["ngram_range"] = list(stored["ngram_range"])
    return {
        "params": stored,
        "dtype": np.dtype(params.get("dtype", np.float64)).name,
        "vocabulary": {str(term): int(index) for term, index in vectorizer.vocabulary_.items()},
    }


def _vectorizer_kwargs(state: dict[str, Any]) -> dict[str, Any]:
    params = dict(state["params"])
    if "ngram_range" in params:
        params["ngram_range"] = tuple(params["ngram_range"])
    if isinstance(params.get("stop_words"), list):
        params["stop_words"] = frozenset(params["stop_words"])
    params["dtype"] = np.dtype(state["dtype"]).type
    params["vocabulary"] = {str(term): int(index) for term, index in state["vocabulary"].items()}
    return params


def restore_count_vectorizer(state: dict[str, Any]) -> CountVectorizer:
    return CountVectorizer(**_vectorizer_kwargs(state))


def restore_tfidf_vectorizer(state: dict[str, Any], idf: np.ndarray | None) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(**_vectorizer_kwargs(state))
    if vectorizer.use_idf:
        if idf is None:
            raise IndexFormatError("TF-IDF vectorizer is missing its IDF weights")
        # The public `idf_` setter exists to transfer fitted IDF weights between vectorizers.
        vectorizer.idf_ = np.asarray(idf, dtype="float64")
    return vectorizer


def pack_matrix(prefix: str, matrix: Any, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Store a dense or CSR/CSC sparse matrix as plain arrays; return its manifest entry."""
    if sparse.issparse(matrix):
        layout = "csc" if matrix.format == "csc" else "csr"
        compressed = matrix.tocsc() if layout == "csc" else matrix.tocsr()
        arrays[f"{prefix}_data"] = compressed.data
        arrays[f"{prefix}_indices"] = compressed.indices
        arrays[f"{prefix}_indptr"] = compressed.indptr
        return {"kind": "sparse", "layout": layout, "shape": [int(size) for size in compressed.shape]}
    arrays[prefix] = np.asarray(matrix)
    return {"kind": "dense"}


def unpack_matrix(prefix: str, entry: dict[str, Any], arrays: dict[str, np.ndarray]) -> Any:
    if entry["kind"] == "dense":
        return arrays[prefix]
    if entry["kind"] != "sparse":
        raise IndexFormatError(f"Unknown matrix kind for {prefix}: {entry['kind']!r}")
    components = (arrays[f"{prefix}_data"], arrays[f"{prefix}_indices"], arrays[f"{prefix}_indptr"])
    shape = tuple(entry["shape"])
    if entry["layout"] == "csc":
        return sparse.csc_matrix(components, shape=shape)
    return sparse.csr_matrix(components, shape=shape)


def write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write the archive to a temp file and rename it, so readers never see half a file."""
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp_path.open("wb") as file_obj:
            np.savez(file_obj, **arrays)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def read_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise IndexFormatError(f"Could not read index file {path}: {exc}") from exc
