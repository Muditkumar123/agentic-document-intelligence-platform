"""Text chunking for document pages."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator

from adip.ingestion.models import Chunk, Page

TOKEN_PATTERN = re.compile(r"\S+")
HEADING_LINE = re.compile(r"#{1,6}\s")
# A line that starts its own block even without a blank line before it: a markdown
# heading, a bullet or numbered list item, or a table row.
STRUCTURAL_LINE = re.compile(r"#{1,6}\s|[-*+•]\s|\d{1,3}[.)]\s|\((?:[a-z]|[ivx]{1,4}|\d{1,2})\)\s|\|")


def count_tokens(text: str) -> int:
    """Return a lightweight whitespace-token count."""
    return len(TOKEN_PATTERN.findall(text))


def normalize_text(text: str) -> str:
    """Normalize parser output while keeping text readable for citations.

    Wrapped lines (a PDF has one per visual line) are joined into one line per
    paragraph. Line breaks are kept where they carry structure: a blank line
    between paragraphs, and a single newline before a heading, list item or
    table row, and after a heading. Other whitespace collapses to one space.
    """
    lines = [" ".join(line.split()) for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    parts: list[str] = []
    current: list[str] = []
    blank_before = False

    def flush() -> None:
        nonlocal blank_before
        if current:
            if parts:
                parts.append("\n\n" if blank_before else "\n")
            parts.append(" ".join(current))
            current.clear()
            blank_before = False

    for line in lines:
        if not line:
            flush()
            blank_before = True
            continue
        if STRUCTURAL_LINE.match(line):
            flush()
        current.append(line)
        if HEADING_LINE.match(line):
            flush()

    flush()
    return "".join(parts)


def split_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> Iterator[tuple[str, int, int]]:
    """Yield word-window chunks as `(text, start_word, end_word)`.

    A chunk's text is the slice of the normalized text from its first word to its
    last, so it keeps the paragraph, heading and list line breaks between them.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    normalized = normalize_text(text)
    words = list(TOKEN_PATTERN.finditer(normalized))
    if not words:
        return

    step = chunk_size - chunk_overlap
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        yield normalized[words[start].start() : words[end - 1].end()], start, end
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
