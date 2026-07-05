"""In-process caches for the API's hot paths.

Two caches, both honest about what they are:

- ``IndexCache`` — the loaded ``RagIndex`` keyed by (path, pickle mtime). Every
  query used to re-unpickle the index from disk; the mtime key means a rebuild
  invalidates the entry naturally, with no explicit invalidation hooks to forget.
- ``QueryCache`` — a bounded LRU over full query responses, keyed by the request
  payload plus the index mtime (so cached answers can never outlive the index
  they were computed from).

This is deliberately a single-process design: it works on the free hosting tier
where no Redis exists, disappears on restart, and is not shared across workers.
Redis is the documented upgrade path when there is more than one instance —
the interface (get/put keyed by a string) is shaped so that swap stays small.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from adip.rag.retriever import INDEX_FILENAME, RagIndex, load_index

DEFAULT_QUERY_CACHE_SIZE = 256
DEFAULT_INDEX_CACHE_SIZE = 4


def index_fingerprint(index_dir: Path) -> tuple[str, int]:
    """(resolved path, mtime_ns) of the index pickle — the invalidation key."""
    pickle_path = index_dir.expanduser() / INDEX_FILENAME
    return str(pickle_path.resolve()), pickle_path.stat().st_mtime_ns


class IndexCache:
    """Keeps recently used RagIndex objects loaded, invalidated by file mtime."""

    def __init__(self, max_entries: int = DEFAULT_INDEX_CACHE_SIZE) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[str, int], RagIndex] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, index_dir: Path) -> RagIndex:
        key = index_fingerprint(index_dir)
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                self.hits += 1
                return self._entries[key]
        index = load_index(index_dir)
        with self._lock:
            self.misses += 1
            self._entries[key] = index
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return index

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "max_entries": self.max_entries,
                "hits": self.hits,
                "misses": self.misses,
            }


class QueryCache:
    """Bounded LRU over serialized responses, keyed by payload + index mtime."""

    def __init__(self, max_entries: int = DEFAULT_QUERY_CACHE_SIZE) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.max_entries = max_entries
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key_for(payload: dict[str, Any], index_dir: Path) -> str:
        path, mtime_ns = index_fingerprint(index_dir)
        material = json.dumps(
            {"payload": payload, "index_path": path, "index_mtime_ns": mtime_ns},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "max_entries": self.max_entries,
                "hits": self.hits,
                "misses": self.misses,
            }


# Module-level singletons: one process, one cache — matching the deployment shape.
INDEX_CACHE = IndexCache()
QUERY_CACHE = QueryCache()


def cache_stats() -> dict[str, Any]:
    return {
        "scope": "in-process (single instance; Redis is the multi-instance upgrade path)",
        "index_cache": INDEX_CACHE.stats(),
        "query_cache": QUERY_CACHE.stats(),
    }
