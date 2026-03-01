"""Ingest orchestrator: build or update the persistent Chroma vector index.

Usage
-----
    # From the semantic_engine/ project root:
    python -m ingest.build_index

    # Smoke-test with first 500 job listings only:
    python -m ingest.build_index --job-limit 500

    # Skip job listings, index only PDFs:
    python -m ingest.build_index --skip-jobs

    # Skip PDFs, index only job listings:
    python -m ingest.build_index --skip-pdfs

The script is safe to re-run: chunks whose chunk_id already exists in Chroma
are skipped (no duplicate embeddings).

Manifest
--------
Each run appends a row to ``storage/manifest.sqlite`` (table ``ingest_runs``)
with counts, timing and any failed documents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Ensure project root is on sys.path when run as __main__ ──────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import (
    CHROMA_COLLECTION,
    CHUNK_OVERLAP_SENTENCES,
    CHUNK_TARGET_TOKENS,
    EMBEDDING_MODEL,
    INGEST_BATCH_SIZE,
    INGEST_VERSION,
    MANIFEST_DB,
    VECTORSTORE_DIR,
)
from ingest.dedup import filter_new_chunks, get_existing_ids
from ingest.ingest_joblistings import ingest_joblistings
from ingest.ingest_pdf import ingest_pdfs


# ── Manifest helpers ──────────────────────────────────────────────────────────

def _init_manifest(db_path: Path) -> None:
    """Create the manifest database and table if they don't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest_runs (
            run_id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ingest_time       TEXT NOT NULL,
            ingest_version    TEXT NOT NULL,
            embedding_model   TEXT NOT NULL,
            chunking_config   TEXT NOT NULL,
            pdf_count         INTEGER NOT NULL DEFAULT 0,
            job_count         INTEGER NOT NULL DEFAULT 0,
            total_chunks      INTEGER NOT NULL DEFAULT 0,
            new_chunks        INTEGER NOT NULL DEFAULT 0,
            skipped_chunks    INTEGER NOT NULL DEFAULT 0,
            failed_docs       TEXT NOT NULL DEFAULT '[]',
            duration_seconds  REAL NOT NULL DEFAULT 0.0
        )
        """
    )
    conn.commit()
    conn.close()


def _write_manifest(
    db_path: Path,
    *,
    pdf_count: int,
    job_count: int,
    total_chunks: int,
    new_chunks: int,
    skipped_chunks: int,
    failed_docs: list[str],
    duration: float,
) -> None:
    chunking_config = json.dumps(
        {"target_tokens": CHUNK_TARGET_TOKENS, "overlap_sentences": CHUNK_OVERLAP_SENTENCES},
        sort_keys=True,
    )
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO ingest_runs
            (ingest_time, ingest_version, embedding_model,
             chunking_config, pdf_count, job_count,
             total_chunks, new_chunks, skipped_chunks,
             failed_docs, duration_seconds)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            INGEST_VERSION,
            EMBEDDING_MODEL,
            chunking_config,
            pdf_count,
            job_count,
            total_chunks,
            new_chunks,
            skipped_chunks,
            json.dumps(failed_docs),
            round(duration, 2),
        ),
    )
    conn.commit()
    conn.close()


# ── Embedding & upsert ────────────────────────────────────────────────────────

def _embed_and_upsert(
    model: SentenceTransformer,
    collection: chromadb.Collection,
    chunks: list[tuple[str, dict]],
    batch_size: int = INGEST_BATCH_SIZE,
) -> None:
    """Embed *chunks* in batches and upsert into *collection*."""
    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        texts = [text for text, _ in batch]
        metas = [meta for _, meta in batch]
        ids = [meta["chunk_id"] for meta in metas]

        embeddings: np.ndarray = model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metas,
        )

        end = min(start + batch_size, total)
        print(f"  upserted {end}/{total} chunks …", file=sys.stderr)


# ── Main ──────────────────────────────────────────────────────────────────────

def build_index(
    skip_pdfs: bool = False,
    skip_jobs: bool = False,
    job_limit: int | None = None,
) -> None:
    t0 = time.monotonic()

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Job RAG — build_index  (version {INGEST_VERSION})", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    # ── Chroma client ─────────────────────────────────────────────────────────
    print(f"[Chroma] Opening persistent store at {VECTORSTORE_DIR}", file=sys.stderr)
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    existing_ids = get_existing_ids(collection)
    print(f"[Chroma] Collection '{CHROMA_COLLECTION}' — {len(existing_ids)} existing chunks", file=sys.stderr)

    # ── Load embedding model ───────────────────────────────────────────────────
    print(f"[Embedder] Loading {EMBEDDING_MODEL} …", file=sys.stderr)
    model = SentenceTransformer(EMBEDDING_MODEL)

    # ── Collect all chunks ────────────────────────────────────────────────────
    all_chunks: list[tuple[str, dict]] = []
    pdf_new = 0
    job_new = 0
    pdf_count = 0
    job_count = 0
    all_failed: list[str] = []

    if not skip_pdfs:
        print("\n[PDF] Ingesting PDF documents …", file=sys.stderr)
        pdf_chunks, pdf_summary = ingest_pdfs()
        pdf_count = pdf_summary["success"]
        all_failed.extend(pdf_summary["failed_docs"])
        new_pdf_chunks, pdf_skipped = filter_new_chunks(pdf_chunks, existing_ids)
        print(
            f"[PDF] {len(new_pdf_chunks)} new, {pdf_skipped} already indexed",
            file=sys.stderr,
        )
        all_chunks.extend(new_pdf_chunks)
        pdf_new = len(new_pdf_chunks)
        # Update existing_ids to prevent collisions within this run
        for _, meta in new_pdf_chunks:
            existing_ids.add(meta["chunk_id"])

    if not skip_jobs:
        print("\n[SQLite] Ingesting job listings …", file=sys.stderr)
        job_chunks, job_summary = ingest_joblistings(limit=job_limit)
        job_count = job_summary["success"]
        all_failed.extend(job_summary["failed_docs"])
        new_job_chunks, job_skipped = filter_new_chunks(job_chunks, existing_ids)
        print(
            f"[SQLite] {len(new_job_chunks)} new, {job_skipped} already indexed",
            file=sys.stderr,
        )
        all_chunks.extend(new_job_chunks)
        job_new = len(new_job_chunks)

    total_new = len(all_chunks)
    total_skipped = (
        (len(pdf_chunks) if not skip_pdfs else 0)
        + (len(job_chunks) if not skip_jobs else 0)
        - total_new
    )

    # ── Embed & upsert ────────────────────────────────────────────────────────
    if all_chunks:
        print(f"\n[Embedder] Embedding & upserting {total_new} new chunks …", file=sys.stderr)
        _embed_and_upsert(model, collection, all_chunks)
    else:
        print("\n[Embedder] Nothing new to index.", file=sys.stderr)

    # ── Manifest ──────────────────────────────────────────────────────────────
    _init_manifest(MANIFEST_DB)
    duration = time.monotonic() - t0
    _write_manifest(
        MANIFEST_DB,
        pdf_count=pdf_count,
        job_count=job_count,
        total_chunks=collection.count(),
        new_chunks=total_new,
        skipped_chunks=total_skipped,
        failed_docs=all_failed[:100],
        duration=duration,
    )

    print(f"\n{'='*60}", file=sys.stderr)
    print(
        f"Done in {duration:.1f}s  |  "
        f"collection size: {collection.count()} chunks  |  "
        f"new: {total_new}  |  skipped: {total_skipped}",
        file=sys.stderr,
    )
    print(f"{'='*60}\n", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or update the Chroma RAG index.")
    parser.add_argument("--skip-pdfs", action="store_true", help="Skip PDF ingestion")
    parser.add_argument("--skip-jobs", action="store_true", help="Skip job listing ingestion")
    parser.add_argument(
        "--job-limit",
        type=int,
        default=None,
        metavar="N",
        help="Only index first N job listings (for smoke-testing)",
    )
    args = parser.parse_args()

    build_index(
        skip_pdfs=args.skip_pdfs,
        skip_jobs=args.skip_jobs,
        job_limit=args.job_limit,
    )
