"""Document parsers for the ingestion pipeline."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

from adip.ingestion.models import Page

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class UnsupportedDocumentError(ValueError):
    """Raised when a document type is not supported."""


class DocumentParseError(RuntimeError):
    """Raised when a supported document cannot be parsed."""


def compute_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_document_id(checksum: str) -> str:
    return f"doc_{checksum[:16]}"


def discover_documents(input_path: Path) -> list[Path]:
    """Return supported document paths from a file or directory."""
    path = input_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise UnsupportedDocumentError(f"Unsupported file extension: {path.suffix}")
        return [path]

    documents = [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(documents)


def parse_document(path: Path) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return parse_text_document(path)
    if suffix == ".pdf":
        return parse_pdf_document(path)
    raise UnsupportedDocumentError(f"Unsupported file extension: {path.suffix}")


def parse_text_document(path: Path) -> list[Page]:
    checksum = compute_checksum(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return [
        Page(
            document_id=build_document_id(checksum),
            source_path=str(path.resolve()),
            filename=path.name,
            source_type=path.suffix.lower().lstrip("."),
            checksum=checksum,
            page_number=1,
            text=text,
            metadata={"parser": "utf8_text"},
        )
    ]


def parse_pdf_document(path: Path) -> list[Page]:
    """Parse a PDF with the system `pdftotext` command."""
    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        raise DocumentParseError(
            "PDF parsing requires the `pdftotext` command or a Python PDF parser."
        )

    checksum = compute_checksum(path)
    command = [pdftotext, "-layout", "-enc", "UTF-8", str(path.resolve()), "-"]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        message = completed.stderr.strip() or "pdftotext failed without stderr output"
        raise DocumentParseError(f"Could not parse PDF {path}: {message}")

    raw_pages = completed.stdout.split("\f")
    pages: list[Page] = []
    document_id = build_document_id(checksum)

    for index, text in enumerate(raw_pages, start=1):
        if not text.strip():
            continue
        pages.append(
            Page(
                document_id=document_id,
                source_path=str(path.resolve()),
                filename=path.name,
                source_type="pdf",
                checksum=checksum,
                page_number=index,
                text=text,
                metadata={"parser": "pdftotext"},
            )
        )

    if not pages:
        raise DocumentParseError(f"No extractable text found in PDF: {path}")

    return pages
