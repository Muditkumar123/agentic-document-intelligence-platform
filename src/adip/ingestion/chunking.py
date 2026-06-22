"""Text chunking for document pages."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator

from adip.ingestion.models import Chunk, Page

TOKEN_PATTERN = re.compile(r"\S+")


def count_tokens(text: str) -> int:
    """Return a lightweight whitespace-token count."""
    return len(TOKEN_PATTERN.findall(text))


def normalize_text(text: str) -> str:
    """Normalize parser output while keeping text readable for citations."""
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    paragraphs: list[str] = []
    current: list[str] = []

    for line in lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line)

    if current:
        paragraphs.append(" ".join(current))

    return "\n\n".join(paragraphs).strip()


def split_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> Iterator[tuple[str, int, int]]:
    """Yield word-window chunks as `(text, start_word, end_word)`."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = TOKEN_PATTERN.findall(normalize_text(text))
    if not words:
        return

    step = chunk_size - chunk_overlap
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        yield " ".join(words[start:end]), start, end
        if end == len(words):
            break
        start += step


def build_chunks(
    pages: Iterable[Page],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Convert parsed pages into traceable chunks."""
    chunks: list[Chunk] = []
    chunk_counters: defaultdict[str, int] = defaultdict(int)

    for page in pages:
        for chunk_text, start_word, end_word in split_text(page.text, chunk_size, chunk_overlap):
            chunk_index = chunk_counters[page.document_id]
            chunk_hash = hashlib.sha1(chunk_text.encode("utf-8")).hexdigest()[:10]
            chunk_id = f"{page.document_id}_p{page.page_number:04d}_c{chunk_index:04d}_{chunk_hash}"
            chunk_counters[page.document_id] += 1

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=page.document_id,
                    filename=page.filename,
                    source_path=page.source_path,
                    source_type=page.source_type,
                    checksum=page.checksum,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    token_count=count_tokens(chunk_text),
                    char_count=len(chunk_text),
                    metadata={
                        **page.metadata,
                        "chunk_strategy": "word_window",
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "start_word": start_word,
                        "end_word": end_word,
                    },
                )
            )

    return chunks
