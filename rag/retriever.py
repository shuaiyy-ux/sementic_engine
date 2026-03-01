"""Chroma retriever: embed a query and return candidate chunks."""

from __future__ import annotations

import sys
import time
from functools import lru_cache
from typing import Any

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import (
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
    RETRIEVE_N,
    VECTORSTORE_DIR,
)


@lru_cache(maxsize=1)
def _get_collection() -> chromadb.Collection:
    """Return the Chroma collection (cached — one client per process)."""
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Return the embedding model (cached — loaded once per process)."""
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_query(text: str) -> np.ndarray:
    """Encode *text* with the shared embedding model."""
    model = _get_model()
    vec: np.ndarray = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vec


def retrieve(
    query: str,
    n_results: int = RETRIEVE_N,
    where: dict[str, Any] | None = None,
) -> tuple[list[dict], float]:
    """Retrieve the top *n_results* chunks most similar to *query*.

    Parameters
    ----------
    query:
        Natural-language user question.
    n_results:
        Number of candidate chunks to return (before re-ranking).
    where:
        Optional Chroma ``where`` filter dict, e.g.
        ``{"source_type": "pdf"}`` or ``{"company": "Acme"}``.

    Returns
    -------
    ``(chunks, elapsed_seconds)``

    Each chunk dict has keys:
        ``id``, ``text``, ``metadata``, ``distance``
        (distances are cosine distances: lower = more similar).
    """
    t0 = time.monotonic()
    collection = _get_collection()

    # Cap n_results to the collection size to avoid Chroma errors on small DBs
    total = collection.count()
    n_results = min(n_results, max(total, 1))

    query_vec = embed_query(query)

    kwargs: dict[str, Any] = {
        "query_embeddings": [query_vec.tolist()],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    try:
        result = collection.query(**kwargs)
    except Exception as exc:
        print(f"[Retriever] Chroma query error: {exc}", file=sys.stderr)
        return [], time.monotonic() - t0

    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]

    chunks = [
        {"id": cid, "text": doc, "metadata": meta, "distance": dist}
        for cid, doc, meta, dist in zip(ids, docs, metas, dists)
    ]

    elapsed = time.monotonic() - t0
    return chunks, elapsed
