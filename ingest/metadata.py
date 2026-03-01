"""Metadata schema and helpers for RAG chunks.

Every chunk ingested into the vector store must carry a ``ChunkMetadata``
dictionary.  All keys are defined here so consuming code can import them from
a single location.
"""

from __future__ import annotations

import hashlib
from typing import TypedDict


class ChunkMetadata(TypedDict):
    """Metadata attached to every chunk stored in Chroma."""

    # Source provenance
    source_type: str        # "pdf" | "job_listing"
    source_id: str          # PDF filename (without path) or job_id as string
    source_uri: str         # Public URL or empty string

    # Human-readable labels
    title: str              # Document title or job title
    company: str            # Company name or empty string
    location: str           # Location string or empty string

    # Page / record anchoring
    page_start: int         # First page of chunk (0 for job_listings)
    page_end: int           # Last page of chunk (0 for job_listings)
    record_id: str          # Original record ID (empty for PDFs)

    # Chunk identity
    chunk_id: str           # Stable unique identifier
    chunk_index: int        # Position within the parent document
    text_hash: str          # SHA-256 hex of chunk text (first 16 chars)

    # Provenance versioning
    ingest_version: str     # Semantic version of ingest pipeline
    embedding_model: str    # HuggingFace model name used to embed this chunk


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_text_hash(text: str) -> str:
    """Return first 16 hex characters of SHA-256(text)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_chunk_id(
    source_type: str,
    source_id: str,
    chunk_index: int,
    text_hash: str,
) -> str:
    """Build a stable, deterministic chunk identifier.

    Format: ``<source_type>_<source_id>_<chunk_index>_<text_hash_prefix>``

    The ``source_id`` is sanitised to remove path separators so the resulting
    ID is safe to use as a filesystem name or Chroma document ID.
    """
    safe_source_id = source_id.replace("/", "-").replace("\\", "-").replace(" ", "_")
    hash_prefix = text_hash[:8]
    return f"{source_type}_{safe_source_id}_{chunk_index}_{hash_prefix}"
