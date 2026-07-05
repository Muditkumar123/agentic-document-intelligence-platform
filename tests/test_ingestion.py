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


def test_table_html_to_text_keeps_cells_attached_to_headers():
    from adip.ingestion.unstructured_parser import table_html_to_text

    html = (
        "<table><tr><th>Class</th><th>Range</th></tr>"
        "<tr><td>Server Error</td><td>5xx</td></tr>"
        "<tr><td>Client Error</td><td>4xx</td></tr></table>"
    )

    text = table_html_to_text(html)

    lines = text.splitlines()
    assert lines[0] == "Class | Range"
    assert "Class: Server Error; Range: 5xx" in lines
    assert "Class: Client Error; Range: 4xx" in lines


def test_table_html_to_text_handles_headerless_and_empty_tables():
    from adip.ingestion.unstructured_parser import table_html_to_text

    assert table_html_to_text("") == ""
    assert table_html_to_text("<p>no table</p>") == ""
    assert table_html_to_text("<table><tr><td>a</td><td>b</td></tr></table>") == "a | b"


def test_element_to_text_prefers_table_html_serialization():
    from adip.ingestion.unstructured_parser import element_to_text

    class FakeMetadata:
        text_as_html = "<table><tr><th>K</th></tr><tr><td>V</td></tr></table>"
        page_number = 1

    class FakeTable:
        category = "Table"
        metadata = FakeMetadata()
        text = "K V flattened"

    class FakeText:
        category = "NarrativeText"
        metadata = FakeMetadata()
        text = "plain paragraph"

    assert element_to_text(FakeTable()) == "K\nK: V"
    assert element_to_text(FakeText()) == "plain paragraph"


def test_parse_document_rejects_unknown_parser(tmp_path):
    from adip.ingestion.parsers import parse_document

    doc = tmp_path / "note.md"
    doc.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported parser"):
        parse_document(doc, parser="mystery")


def test_unstructured_parser_requires_the_extra(tmp_path):
    import importlib.util

    from adip.ingestion.unstructured_parser import parse_document_unstructured

    if importlib.util.find_spec("unstructured") is not None:
        pytest.skip("unstructured installed; lazy-import guard not exercisable")
    doc = tmp_path / "note.md"
    doc.write_text("hello", encoding="utf-8")
    with pytest.raises(ImportError, match=r"pip install -e \"\.\[tables\]\""):
        parse_document_unstructured(doc)


def test_parse_document_unstructured_assembles_pages_and_counts_tables(tmp_path, monkeypatch):
    import adip.ingestion.unstructured_parser as up

    class FakeMeta:
        def __init__(self, page, html=None):
            self.page_number = page
            self.text_as_html = html

    class FakeElement:
        def __init__(self, category, text, page, html=None):
            self.category = category
            self.text = text
            self.metadata = FakeMeta(page, html)

    elements = [
        FakeElement("Title", "Status Codes", 1),
        FakeElement(
            "Table",
            "flattened",
            1,
            html="<table><tr><th>Class</th><th>Range</th></tr><tr><td>Server Error</td><td>5xx</td></tr></table>",
        ),
        FakeElement("NarrativeText", "404 means not found.", 2),
    ]
    monkeypatch.setattr(up, "unstructured_available", lambda: True)

    import sys
    import types

    fake_auto = types.ModuleType("unstructured.partition.auto")
    fake_auto.partition = lambda filename: elements
    fake_partition = types.ModuleType("unstructured.partition")
    fake_root = types.ModuleType("unstructured")
    monkeypatch.setitem(sys.modules, "unstructured", fake_root)
    monkeypatch.setitem(sys.modules, "unstructured.partition", fake_partition)
    monkeypatch.setitem(sys.modules, "unstructured.partition.auto", fake_auto)

    doc = tmp_path / "codes.pdf"
    doc.write_bytes(b"%PDF-1.4 fake")

    pages = up.parse_document_unstructured(doc)

    assert len(pages) == 2
    assert pages[0].metadata["parser"] == "unstructured"
    assert pages[0].metadata["table_count"] == 1
    assert "Class: Server Error; Range: 5xx" in pages[0].text
    assert pages[1].metadata["table_count"] == 0
    assert pages[1].text == "404 means not found."


def test_ingest_path_passes_parser_through(tmp_path):
    doc_dir = tmp_path / "raw"
    doc_dir.mkdir()
    (doc_dir / "note.md").write_text("# Title\n\nBody text here.", encoding="utf-8")

    result = ingest_path(doc_dir, tmp_path / "chunks.jsonl", parser="default")

    assert result.parser == "default"
    assert result.chunk_count >= 1
