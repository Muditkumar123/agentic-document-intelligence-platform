"""Qdrant vector store for dense and hybrid indexes.

``vector_store="qdrant"`` keeps an index's embeddings in Qdrant instead of in the
index archive, and runs dense search there: approximate nearest neighbours with
the document filter pushed down as a point-id filter, the same code path whether
Qdrant runs embedded or as a server.

- **Embedded** (default): Qdrant's local mode stores the collection in
  ``<index dir>/qdrant``. No server, nothing to deploy, but it is single-process:
  the folder is locked while a client has it open.
- **Server**: set ``ADIP_QDRANT_URL`` (and ``ADIP_QDRANT_API_KEY`` if needed) and the
  collection lives on that server, named after the index directory.

Point ids are chunk row numbers, so a hit maps straight back to ``index.chunks``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

QDRANT_URL_ENV = "ADIP_QDRANT_URL"
QDRANT_API_KEY_ENV = "ADIP_QDRANT_API_KEY"
QDRANT_DIRNAME = "qdrant"
COLLECTION = "chunks"
UPLOAD_BATCH = 256


def _require_qdrant() -> None:
    if importlib.util.find_spec("qdrant_client") is None:
        raise ImportError('qdrant-client is required for vector_store="qdrant": pip install -e ".[vectordb]"')


def server_url() -> str | None:
    return os.environ.get(QDRANT_URL_ENV) or None


def collection_for(index_dir: Path) -> str:
    """Server-side collection name: stable per index directory."""
    resolved = str(index_dir.expanduser().resolve())
    return f"adip_{index_dir.name}_{hashlib.sha256(resolved.encode()).hexdigest()[:10]}"


class QdrantVectorStore:
    def __init__(self, client: Any, collection: str, location: str) -> None:
        self.client = client
        self.collection = collection
        self.location = location

    @classmethod
    def _client(cls, location: str) -> Any:
        _require_qdrant()
        from qdrant_client import QdrantClient

        if location == ":memory:":
            return QdrantClient(location=":memory:")
        if location.startswith(("http://", "https://")):
            return QdrantClient(url=location, api_key=os.environ.get(QDRANT_API_KEY_ENV) or None)
        return QdrantClient(path=location)

    @classmethod
    def create(cls, location: str, vectors: np.ndarray, payloads: list[dict[str, Any]], collection: str = COLLECTION):
        """Create (or replace) a collection holding one point per row of ``vectors``."""
        from qdrant_client import models

        if location not in {":memory:"} and not location.startswith(("http://", "https://")):
            shutil.rmtree(location, ignore_errors=True)
            Path(location).mkdir(parents=True, exist_ok=True)
        client = cls._client(location)
        if client.collection_exists(collection):
            client.delete_collection(collection)
        client.create_collection(
            collection,
            vectors_config=models.VectorParams(size=int(vectors.shape[1]), distance=models.Distance.COSINE),
        )
        for start in range(0, len(vectors), UPLOAD_BATCH):
            client.upsert(
                collection,
                points=[
                    models.PointStruct(id=row, vector=vectors[row].tolist(), payload=payloads[row])
                    for row in range(start, min(start + UPLOAD_BATCH, len(vectors)))
                ],
            )
        return cls(client, collection, location)

    @classmethod
    def open(cls, location: str, collection: str = COLLECTION):
        client = cls._client(location)
        if not client.collection_exists(collection):
            raise FileNotFoundError(f"Qdrant collection {collection!r} not found at {location}")
        return cls(client, collection, location)

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    def search(self, vector: np.ndarray, limit: int, allowed_rows: list[int] | None = None) -> list[tuple[int, float]]:
        """``(row, cosine score)`` for the nearest points, best first, optionally only among ``allowed_rows``."""
        from qdrant_client import models

        if limit <= 0 or allowed_rows == []:
            return []
        query_filter = None
        if allowed_rows is not None:
            query_filter = models.Filter(must=[models.HasIdCondition(has_id=list(allowed_rows))])
        response = self.client.query_points(
            self.collection,
            query=np.asarray(vector, dtype="float32").ravel().tolist(),
            limit=limit,
            query_filter=query_filter,
        )
        return [(int(point.id), float(point.score)) for point in response.points]

    def vectors(self) -> np.ndarray:
        """All stored vectors in row order (used when re-saving a loaded index elsewhere)."""
        rows: dict[int, list[float]] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=UPLOAD_BATCH, offset=offset, with_vectors=True, with_payload=False
            )
            for point in points:
                rows[int(point.id)] = point.vector
            if offset is None:
                break
        return np.asarray([rows[row] for row in sorted(rows)], dtype="float32")

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # closing is best-effort; the process may already be exiting
            pass


def chunk_payload(chunk: dict[str, Any]) -> dict[str, Any]:
    return {key: chunk.get(key) for key in ("chunk_id", "document_id", "filename", "page_number")}
