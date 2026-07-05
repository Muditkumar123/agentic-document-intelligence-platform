"""Optional unstructured.io parser with table extraction.

The default parsers flatten document structure: ``pdftotext -layout`` renders a
PDF table as loosely aligned whitespace, which chunks poorly and retrieves
worse. unstructured.io partitions a document into typed elements — including
``Table`` elements with an HTML rendering — so tables can be serialized into
retrieval-friendly text where every cell stays attached to its column header.

unstructured is heavy (its PDF path alone pulls pdfminer and friends), so it
lives behind the ``[tables]`` extra and is imported lazily; the default
ingestion path never touches it. The HTML-table-to-text serialization is
dependency-free stdlib and is what the tests exercise directly.
"""

from __future__ import annotations

import importlib.util
from html.parser import HTMLParser
from pathlib import Path

from adip.ingestion.models import Page


def unstructured_available() -> bool:
    return importlib.util.find_spec("unstructured") is not None


class _TableHTMLParser(HTMLParser):
    """Collects an HTML table into a list of rows of cell strings."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"}:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def table_html_to_text(html: str) -> str:
    """Serialize an HTML table so each cell stays attached to its column header.

    The first row is treated as the header; every following row becomes one
    line of ``header: value`` pairs. This keeps "which column does this number
    belong to" recoverable by a lexical retriever, which a flattened layout
    loses. Falls back to pipe-joined rows when there is no header row.
    """
    parser = _TableHTMLParser()
    parser.feed(html or "")
    rows = parser.rows
    if not rows:
        return ""
    if len(rows) == 1:
        return " | ".join(rows[0])

    header = rows[0]
    lines: list[str] = [" | ".join(header)]
    for row in rows[1:]:
        if len(row) == len(header):
            lines.append("; ".join(f"{name}: {value}" for name, value in zip(header, row)))
        else:
            lines.append(" | ".join(row))
    return "\n".join(lines)


def element_to_text(element) -> str:
    """Best text rendering of an unstructured element (tables get header-aware
    serialization from their HTML when available)."""
    category = getattr(element, "category", None) or type(element).__name__
    if category == "Table":
        html = getattr(getattr(element, "metadata", None), "text_as_html", None)
        if html:
            serialized = table_html_to_text(html)
            if serialized:
                return serialized
    return str(getattr(element, "text", "") or "")


def parse_document_unstructured(path: Path) -> list[Page]:
    """Partition a document with unstructured.io and assemble per-page text."""
    if not unstructured_available():
        raise ImportError(
            'unstructured is required for the unstructured parser. Install the extra: pip install -e ".[tables]"'
        )

    from unstructured.partition.auto import partition

    from adip.ingestion.parsers import build_document_id, compute_checksum

    elements = partition(filename=str(path.expanduser().resolve()))
    checksum = compute_checksum(path)
    document_id = build_document_id(checksum)

    page_texts: dict[int, list[str]] = {}
    page_table_counts: dict[int, int] = {}
    for element in elements:
        page_number = getattr(getattr(element, "metadata", None), "page_number", None) or 1
        text = element_to_text(element).strip()
        if not text:
            continue
        page_texts.setdefault(page_number, []).append(text)
        if (getattr(element, "category", None) or type(element).__name__) == "Table":
            page_table_counts[page_number] = page_table_counts.get(page_number, 0) + 1

    pages = [
        Page(
            document_id=document_id,
            source_path=str(path.resolve()),
            filename=path.name,
            source_type=path.suffix.lower().lstrip("."),
            checksum=checksum,
            page_number=page_number,
            text="\n\n".join(texts),
            metadata={
                "parser": "unstructured",
                "table_count": page_table_counts.get(page_number, 0),
            },
        )
        for page_number, texts in sorted(page_texts.items())
    ]
    if not pages:
        raise ValueError(f"No extractable content found in document: {path}")
    return pages
