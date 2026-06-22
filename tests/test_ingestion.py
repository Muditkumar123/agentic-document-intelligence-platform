import json

import pytest

from adip.ingestion.chunking import split_text
from adip.ingestion.pipeline import ingest_path


def test_split_text_uses_overlap():
    text = " ".join(f"word{i}" for i in range(12))

    chunks = list(split_text(text, chunk_size=5, chunk_overlap=2))

    assert [chunk[1:] for chunk in chunks] == [(0, 5), (3, 8), (6, 11), (9, 12)]
    assert chunks[0][0] == "word0 word1 word2 word3 word4"
    assert chunks[1][0].startswith("word3 word4")


def test_split_text_rejects_invalid_overlap():
    with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
        list(split_text("hello world", chunk_size=4, chunk_overlap=4))


def test_ingest_text_directory_writes_jsonl(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source_file = raw_dir / "sample.txt"
    source_file.write_text(
        "This is a sample document for ingestion. "
        "It has enough words to create multiple chunks with overlap. "
        "The pipeline should preserve source metadata and write JSONL.",
        encoding="utf-8",
    )
    output_path = tmp_path / "processed" / "chunks.jsonl"

    result = ingest_path(raw_dir, output_path, chunk_size=8, chunk_overlap=2)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]

    assert result.document_count == 1
    assert result.page_count == 1
    assert result.chunk_count == len(records)
    assert result.chunk_count > 1
    assert records[0]["filename"] == "sample.txt"
    assert records[0]["source_type"] == "txt"
    assert records[0]["page_number"] == 1
    assert records[0]["metadata"]["chunk_strategy"] == "word_window"
