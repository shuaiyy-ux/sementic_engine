"""Deduplication utilities for the ingest pipeline.

A chunk is considered a duplicate if its ``chunk_id`` already exists in the
Chroma collection.  Because ``chunk_id`` encodes ``source_type``,
``source_id``, ``chunk_index`` *and* a hash prefix of the chunk text, any
change to the source document will produce new chunk IDs and therefore trigger
re-embedding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import chromadb


def get_existing_ids(collection: "chromadb.Collection") -> set[str]:
    """Return the set of all chunk IDs already stored in *collection*.

    Uses ``collection.get`` with ``include=[]`` to avoid fetching embeddings
    or documents — this is much cheaper for large collections.
    """
    try:
        result = collection.get(include=[])
        return set(result["ids"])
    except Exception:
        return set()


def filter_new_chunks(
    chunks: list[tuple[str, dict]],
    existing_ids: set[str],
) -> tuple[list[tuple[str, dict]], int]:
    """Split *chunks* into (new_chunks, skipped_count).

    A chunk is new if its ``chunk_id`` metadata key is NOT in *existing_ids*.
    """
    new: list[tuple[str, dict]] = []
    skipped = 0

    for text, meta in chunks:
        if meta["chunk_id"] in existing_ids:
            skipped += 1
        else:
            new.append((text, meta))

    return new, skipped
