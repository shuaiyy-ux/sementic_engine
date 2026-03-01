"""Cross-encoder reranker (Advanced Feature: Re-ranking).

Loads ``cross-encoder/ms-marco-MiniLM-L-6-v2`` and scores each (query, chunk)
pair.  The top-K chunks after reranking are returned together with a log of
rank changes for technical writeup evidence.

The reranker is lazy-loaded and cached so the model is only downloaded once.
"""

from __future__ import annotations

import sys
import time
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import FINAL_K, RETRIEVE_N, RERANKER_MODEL


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    """Load and cache the cross-encoder model."""
    print(f"[Reranker] Loading {RERANKER_MODEL} …", file=sys.stderr)
    return CrossEncoder(RERANKER_MODEL, max_length=512)


def rerank(
    query: str,
    chunks: list[dict],
    top_k: int = FINAL_K,
) -> tuple[list[dict], float, list[dict]]:
    """Score *chunks* against *query* and return the top *top_k* after reranking.

    Parameters
    ----------
    query:
        User question (may already be rewritten for follow-ups).
    chunks:
        Candidate chunks from the vector retriever (list of dicts with at
        least ``"text"`` and ``"id"`` keys).
    top_k:
        Number of chunks to keep after scoring.

    Returns
    -------
    ``(reranked_chunks, elapsed_seconds, rank_log)``

    ``rank_log`` is a list of dicts recording each chunk's original rank and
    new rank — useful for the technical writeup.
    """
    if not chunks:
        return [], 0.0, []

    t0 = time.monotonic()
    reranker = _get_reranker()

    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores: list[float] = reranker.predict(pairs, show_progress_bar=False).tolist()

    # Attach scores and sort
    for i, (chunk, score) in enumerate(zip(chunks, scores)):
        chunk["rerank_score"] = score
        chunk["original_rank"] = i + 1  # 1-based

    sorted_chunks = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)

    # Build rank log
    rank_log = []
    for new_rank, chunk in enumerate(sorted_chunks, start=1):
        rank_log.append(
            {
                "chunk_id": chunk["id"],
                "original_rank": chunk["original_rank"],
                "new_rank": new_rank,
                "rerank_score": round(chunk["rerank_score"], 4),
                "distance": round(chunk.get("distance", 0.0), 4),
                "title": chunk.get("metadata", {}).get("title", ""),
            }
        )

    top_chunks = sorted_chunks[:top_k]
    elapsed = time.monotonic() - t0

    # Log summary
    print(
        f"[Reranker] Reranked {len(chunks)} → {len(top_chunks)} chunks in {elapsed:.2f}s",
        file=sys.stderr,
    )
    for entry in rank_log[:top_k]:
        arrow = (
            "↑" if entry["new_rank"] < entry["original_rank"]
            else ("↓" if entry["new_rank"] > entry["original_rank"] else "=")
        )
        print(
            f"  {arrow}  orig #{entry['original_rank']:2d} → new #{entry['new_rank']:2d}  "
            f"score {entry['rerank_score']:+.3f}  [{entry['title'][:40]}]",
            file=sys.stderr,
        )

    return top_chunks, elapsed, rank_log
