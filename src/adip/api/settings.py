"""Server-side configuration for the HTTP API.

Every filesystem location the API reads or writes is decided here, never by a
request: clients pick a retrieval backend by name and the server maps it to a
path. Earlier versions accepted `index_path`, `raw_dir`, `input_path` and
`trace_dir` from callers, which let anyone who could reach the API write uploads,
build indexes, delete documents and write traces anywhere the process could.

Environment variables (a local `.env` works too):

- ``ADIP_ADMIN_TOKEN`` — bearer token required by the write endpoints (upload,
  delete, rebuild). Unset means writes are only accepted from loopback clients.
- ``ADIP_MAX_UPLOAD_MB`` — largest accepted document upload (default 20).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from adip.config.env import load_project_env

ADMIN_TOKEN_ENV = "ADIP_ADMIN_TOKEN"
MAX_UPLOAD_MB_ENV = "ADIP_MAX_UPLOAD_MB"
DEFAULT_MAX_UPLOAD_MB = 20
# Headroom over the file size for multipart boundaries and form fields.
MULTIPART_OVERHEAD_BYTES = 1024 * 1024

IndexSlot = Literal["tfidf", "dense", "hybrid"]
INDEX_SLOT_SUFFIXES: dict[str, str] = {"tfidf": "", "dense": "_dense", "hybrid": "_hybrid"}
INDEX_SLOTS: tuple[IndexSlot, ...] = ("tfidf", "dense", "hybrid")


def index_slot_for_backend(backend: str) -> IndexSlot:
    """Which of the three standard index directories a retrieval backend is stored in."""
    if backend in {"tfidf", "hybrid"}:
        return backend  # type: ignore[return-value]
    if backend in {"dense", "dense_lsa", "sentence_transformers"}:
        return "dense"
    raise ValueError(f"Unsupported backend: {backend}")


@dataclass(frozen=True)
class ApiSettings:
    data_dir: Path = Path("data")
    admin_token: str | None = None
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_MB * 1024 * 1024

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "processed" / "chunks.jsonl"

    @property
    def trace_dir(self) -> Path:
        return self.data_dir / "monitoring" / "agent_traces"

    @property
    def mlops_run_dir(self) -> Path:
        return self.data_dir / "monitoring" / "mlops_runs"

    @property
    def max_request_bytes(self) -> int:
        return self.max_upload_bytes + MULTIPART_OVERHEAD_BYTES

    def index_path(self, backend: str = "tfidf") -> Path:
        slot = index_slot_for_backend(backend)
        return self.data_dir / "processed" / f"vector_index{INDEX_SLOT_SUFFIXES[slot]}"

    @property
    def write_access(self) -> str:
        return "admin_token" if self.admin_token else "local_only"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ApiSettings:
        if environ is None:
            load_project_env()
            environ = os.environ
        token = (environ.get(ADMIN_TOKEN_ENV) or "").strip() or None
        raw_limit = (environ.get(MAX_UPLOAD_MB_ENV) or "").strip()
        try:
            max_upload_mb = float(raw_limit) if raw_limit else DEFAULT_MAX_UPLOAD_MB
        except ValueError as exc:
            raise ValueError(f"{MAX_UPLOAD_MB_ENV} must be a number, got {raw_limit!r}") from exc
        if max_upload_mb <= 0:
            raise ValueError(f"{MAX_UPLOAD_MB_ENV} must be greater than 0")
        return cls(admin_token=token, max_upload_bytes=int(max_upload_mb * 1024 * 1024))
