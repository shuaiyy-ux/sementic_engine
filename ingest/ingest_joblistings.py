"""Job listing ingestion: read SQLite → chunk → return chunk list.

Reads from the ``job_listings_cleaned`` table in ``linkedin_jobs_cleaned.sqlite``
and converts each row into sentence-level chunks.

The SQLite table columns are:
    job_id, company_id, title, clean_title, description

Because the table does not store company name strings (only ``company_id``),
the ``company`` metadata field is left empty for job listing chunks.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

from app.config import SQLITE_DB
from ingest.chunking import chunk_text

_TABLE = "job_listings_cleaned"
_DATASET_URI = "https://huggingface.co/datasets/Fhujnfjfj/linkedin-jobs-sqlite"


def ingest_joblistings(
    db_path: Path | None = None,
    limit: int | None = None,
) -> tuple[list[tuple[str, dict]], dict]:
    """Read job listings from SQLite and return a flat list of chunks.

    Parameters
    ----------
    db_path:
        Path to the SQLite database.  Defaults to ``SQLITE_DB`` from config.
    limit:
        If set, only process the first *limit* rows (useful for smoke-testing).

    Returns
    -------
    ``(all_chunks, summary)`` where summary contains counts.
    """
    db_path = db_path or SQLITE_DB

    if not db_path.exists():
        print(
            f"[WARN] SQLite DB not found at {db_path}. Skipping job listing ingest.",
            file=sys.stderr,
        )
        return [], {"total_jobs": 0, "success": 0, "failed_docs": [], "total_chunks": 0}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    query = f"SELECT job_id, title, description FROM {_TABLE}"
    if limit:
        query += f" LIMIT {limit}"

    try:
        rows = conn.execute(query).fetchall()
    except Exception as exc:
        print(f"[ERROR] Failed to query {_TABLE}: {exc}", file=sys.stderr)
        conn.close()
        return [], {"total_jobs": 0, "success": 0, "failed_docs": [], "total_chunks": 0}
    finally:
        conn.close()

    all_chunks: list[tuple[str, dict]] = []
    failed: list[str] = []
    success_count = 0

    for row in rows:
        job_id = str(row["job_id"])
        title = (row["title"] or "").strip() or "Untitled Position"
        description = (row["description"] or "").strip()

        if not description:
            failed.append(job_id)
            continue

        # Combine title + description as the document text
        full_text = f"{title}\n\n{description}"

        chunks = chunk_text(
            text=full_text,
            source_type="job_listing",
            source_id=job_id,
            title=title,
            company="",       # company_id only, no name string in schema
            location="",
            source_uri=_DATASET_URI,
            page_map=None,    # no page structure for listings
            record_id=job_id,
        )

        if chunks:
            all_chunks.extend(chunks)
            success_count += 1
        else:
            failed.append(job_id)

    summary = {
        "total_jobs": len(rows),
        "success": success_count,
        "failed_docs": failed[:50],  # cap list length for manifest storage
        "total_chunks": len(all_chunks),
    }

    print(
        f"[SQLite] {success_count}/{len(rows)} job listings processed → "
        f"{len(all_chunks)} chunks",
        file=sys.stderr,
    )
    return all_chunks, summary
