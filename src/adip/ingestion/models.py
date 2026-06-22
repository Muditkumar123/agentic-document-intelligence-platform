"""Data models for ingestion outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Page:
    document_id: str
    source_path: str
    filename: str
    source_type: str
    checksum: str
    page_number: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    filename: str
    source_path: str
    source_type: str
    checksum: str
    page_number: int
    chunk_index: int
    text: str
    token_count: int
    char_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
