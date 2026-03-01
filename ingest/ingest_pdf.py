"""PDF ingestion: text extraction and chunking for all PDF files.

Uses ``pdfplumber`` for text extraction.  Scanned PDFs (no selectable text)
are skipped with a warning logged to stderr.

Naming convention for titles:
    ``NNN_<slug>.pdf``  →  strip leading digits + underscore, replace hyphens
    with spaces, then title-case.  E.g. ``025_senior-programmer-analyst.pdf``
    becomes ``"Senior Programmer Analyst"``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pdfplumber

from app.config import EMBEDDING_MODEL, INGEST_VERSION, PDF_DIR
from ingest.chunking import chunk_text


def _derive_title(filename: str) -> str:
    """Convert a PDF filename into a human-readable title."""
    stem = Path(filename).stem          # remove .pdf
    stem = re.sub(r"^\d+_", "", stem)   # strip leading NNN_
    stem = stem.replace("-pdf", "")     # strip trailing -pdf suffix
    stem = stem.replace("-", " ").replace("_", " ")
    return stem.title()


def load_pdf(path: Path) -> list[tuple[int, int, str]]:
    """Extract text page by page.

    Returns:
        List of ``(page_number_1based, page_number_1based, page_text)`` tuples.
        Returns an empty list if the PDF has no selectable text.
    """
    pages: list[tuple[int, int, str]] = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                text = text.strip()
                if text:
                    pages.append((i, i, text))
    except Exception as exc:
        print(f"[WARN] Could not read {path.name}: {exc}", file=sys.stderr)
    return pages


def ingest_pdfs(
    pdf_dir: Path | None = None,
) -> list[tuple[str, dict]]:
    """Process all PDFs in *pdf_dir* and return a flat list of chunks.

    Parameters
    ----------
    pdf_dir:
        Directory containing PDF files.  Defaults to ``PDF_DIR`` from config.

    Returns
    -------
    List of ``(chunk_text, ChunkMetadata)`` tuples ready for embedding.
    Also returns a summary dict as a second return value.
    """
    pdf_dir = pdf_dir or PDF_DIR
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    all_chunks: list[tuple[str, dict]] = []
    failed: list[str] = []
    success_count = 0
    empty_count = 0

    for pdf_path in pdf_files:
        page_map = load_pdf(pdf_path)

        if not page_map:
            print(f"[SKIP] {pdf_path.name} — no extractable text", file=sys.stderr)
            empty_count += 1
            failed.append(pdf_path.name)
            continue

        full_text = "\n\n".join(page_text for _, _, page_text in page_map)
        title = _derive_title(pdf_path.name)

        chunks = chunk_text(
            text=full_text,
            source_type="pdf",
            source_id=pdf_path.name,
            title=title,
            company="",
            location="",
            source_uri="",
            page_map=page_map,
            record_id="",
        )

        if chunks:
            all_chunks.extend(chunks)
            success_count += 1
        else:
            empty_count += 1
            failed.append(pdf_path.name)

    summary = {
        "total_pdfs": len(pdf_files),
        "success": success_count,
        "empty_or_failed": empty_count,
        "failed_docs": failed,
        "total_chunks": len(all_chunks),
    }

    print(
        f"[PDF] {success_count}/{len(pdf_files)} PDFs processed → "
        f"{len(all_chunks)} chunks",
        file=sys.stderr,
    )
    return all_chunks, summary
