"""Chunk loading helpers for RAG."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_CHUNK_FIELDS = {
    "chunk_id",
    "document_id",
    "filename",
    "source_path",
    "source_type",
    "checksum",
    "page_number",
    "chunk_index",
    "text",
    "token_count",
    "char_count",
    "metadata",
}


def read_chunks_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load chunk records from an ingestion JSONL file."""
    chunks: list[dict[str, Any]] = []
    with path.expanduser().open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            validate_chunk_record(record, path, line_number)
            chunks.append(record)

    if not chunks:
        raise ValueError(f"No chunk records found in {path}")
    return chunks


def validate_chunk_record(record: dict[str, Any], path: Path, line_number: int) -> None:
    missing = REQUIRED_CHUNK_FIELDS.difference(record)
    if missing:
        missing_fields = ", ".join(sorted(missing))
        raise ValueError(f"{path}:{line_number} is missing required fields: {missing_fields}")

    if not isinstance(record["text"], str) or not record["text"].strip():
        raise ValueError(f"{path}:{line_number} has empty chunk text")
