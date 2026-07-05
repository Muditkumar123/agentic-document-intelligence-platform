"""End-to-end ingestion pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from adip.ingestion.chunking import build_chunks
from adip.ingestion.models import Chunk, Page
from adip.ingestion.parsers import discover_documents, parse_document


@dataclass(frozen=True)
class IngestionResult:
    input_path: str
    output_path: str
    document_count: int
    page_count: int
    chunk_count: int
    chunk_size: int
    chunk_overlap: int
    parser: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ingest_path(
    input_path: Path,
    output_path: Path,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    parser: str = "default",
) -> IngestionResult:
    """Parse supported documents and write chunks to JSONL."""
    documents = discover_documents(input_path)
    if not documents:
        raise FileNotFoundError(f"No supported documents found under: {input_path}")

    pages: list[Page] = []
    for document in documents:
        pages.extend(parse_document(document, parser=parser))

    chunks = build_chunks(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    write_chunks_jsonl(chunks, output_path)

    return IngestionResult(
        input_path=str(input_path),
        output_path=str(output_path),
        document_count=len(documents),
        page_count=len(pages),
        chunk_count=len(chunks),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        parser=parser,
    )


def write_chunks_jsonl(chunks: list[Chunk], output_path: Path) -> None:
    path = output_path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for chunk in chunks:
            file_obj.write(json.dumps(chunk.to_dict(), ensure_ascii=False, sort_keys=True))
            file_obj.write("\n")
