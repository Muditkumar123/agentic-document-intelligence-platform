"""Classification dataset built from the eval corpus with document-level splits."""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from adip.ingestion.chunking import split_text
from adip.ingestion.parsers import discover_documents, parse_document

SOURCES_ROW_PATTERN = re.compile(r"^\|\s*`(?P<filename>[^`]+)`\s*\|\s*(?P<category>[a-z]+)\s*\|")


@dataclass(frozen=True)
class LabeledChunk:
    text: str
    label: str
    filename: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_sources_categories(sources_path: Path) -> dict[str, str]:
    """filename -> category from the SOURCES.md table (the authoritative labels)."""
    categories: dict[str, str] = {}
    for line in sources_path.read_text(encoding="utf-8").splitlines():
        match = SOURCES_ROW_PATTERN.match(line.strip())
        if match:
            categories[match.group("filename")] = match.group("category")
    if not categories:
        raise ValueError(f"No document categories found in {sources_path}")
    return categories


def build_labeled_chunks(
    raw_dir: Path,
    categories: dict[str, str],
    chunk_size: int = 40,
    chunk_overlap: int = 0,
    min_chars: int = 80,
) -> list[LabeledChunk]:
    """Chunk every categorized document at classification granularity.

    ``chunk_size`` counts words (the ingestion chunker is a word window).
    Retrieval chunks (800 words) are far too coarse for a useful sample count
    on this corpus, so the dataset re-chunks at ~40 words and drops fragments
    below ``min_chars``. Overlap is 0 so no two training samples share text.
    """
    labeled: list[LabeledChunk] = []
    for document in discover_documents(raw_dir):
        category = categories.get(document.name)
        if category is None:
            continue
        for page in parse_document(document):
            for text, _start, _end in split_text(
                page.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            ):
                cleaned = " ".join(text.split())
                if len(cleaned) >= min_chars:
                    labeled.append(LabeledChunk(text=cleaned, label=category, filename=document.name))
    if not labeled:
        raise ValueError(f"No labeled chunks produced from {raw_dir}")
    return labeled


def split_by_document(
    chunks: list[LabeledChunk],
    holdout_per_category: int = 1,
    seed: int = 13,
) -> tuple[list[LabeledChunk], list[LabeledChunk]]:
    """Document-level train/eval split.

    Chunks from one document never appear on both sides — a chunk-level split
    would leak near-identical neighbouring text into eval and inflate accuracy.
    """
    if holdout_per_category < 1:
        raise ValueError("holdout_per_category must be at least 1")
    by_category: dict[str, list[str]] = {}
    for chunk in chunks:
        filenames = by_category.setdefault(chunk.label, [])
        if chunk.filename not in filenames:
            filenames.append(chunk.filename)

    rng = random.Random(seed)
    eval_documents: set[str] = set()
    for category, filenames in sorted(by_category.items()):
        if len(filenames) <= holdout_per_category:
            raise ValueError(
                f"Category {category!r} has only {len(filenames)} document(s); "
                "cannot hold out a full document and still train on it"
            )
        eval_documents.update(rng.sample(sorted(filenames), holdout_per_category))

    train = [chunk for chunk in chunks if chunk.filename not in eval_documents]
    evaluation = [chunk for chunk in chunks if chunk.filename in eval_documents]
    return train, evaluation
