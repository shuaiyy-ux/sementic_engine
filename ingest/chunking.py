"""Sentence-level chunking with token-window grouping and sentence overlap.

Strategy
--------
1. Split the source text into sentences using ``nltk.sent_tokenize``.
2. Group consecutive sentences into chunks whose token length stays close to
   ``CHUNK_TARGET_TOKENS`` (measured with ``tiktoken``).
3. Slide the window forward by at most ``CHUNK_OVERLAP_SENTENCES`` sentences,
   so adjacent chunks share context.

The function is deterministic: identical (text, config) inputs always produce
the same chunk list with the same chunk IDs.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterator

import nltk
import tiktoken

from app.config import (
    CHUNK_OVERLAP_SENTENCES,
    CHUNK_TARGET_TOKENS,
    EMBEDDING_MODEL,
    INGEST_VERSION,
)
from ingest.metadata import ChunkMetadata, build_chunk_id, compute_text_hash

# Ensure the NLTK sentence tokeniser data is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# Use the same tokeniser as the LLM (cl100k_base is a reasonable default for
# counting tokens even when the embedding model uses a different tokeniser)
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, stripping blank lines first."""
    # Collapse excessive whitespace / blank lines to aid sentence splitting
    cleaned = re.sub(r"\n{2,}", "\n\n", text).strip()
    sentences = nltk.sent_tokenize(cleaned)
    # Remove trivially short fragments (e.g. single characters)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def chunk_text(
    text: str,
    source_type: str,
    source_id: str,
    title: str,
    company: str = "",
    location: str = "",
    source_uri: str = "",
    page_map: list[tuple[int, int, str]] | None = None,
    record_id: str = "",
) -> list[tuple[str, ChunkMetadata]]:
    """Split *text* into overlapping sentence-window chunks.

    Parameters
    ----------
    text:
        Full document text to chunk.
    source_type:
        ``"pdf"`` or ``"job_listing"``.
    source_id:
        PDF filename (no path) or job_id as string.
    title:
        Document / job title.
    company:
        Company name (or empty string).
    location:
        Location (or empty string).
    source_uri:
        Public URL or empty string.
    page_map:
        Optional list of ``(page_start, page_end, page_text)`` tuples from the
        PDF loader.  When supplied, each sentence is mapped back to the page(s)
        it came from.
    record_id:
        Original SQLite record ID (for job_listing sources).

    Returns
    -------
    List of ``(chunk_text, ChunkMetadata)`` tuples.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    # Build sentence → page_number mapping (for PDF sources)
    sentence_pages: list[int] = []
    if page_map:
        for sent in sentences:
            page = _find_sentence_page(sent, page_map)
            sentence_pages.append(page)
    else:
        sentence_pages = [0] * len(sentences)

    chunks: list[tuple[str, ChunkMetadata]] = []
    start_idx = 0

    while start_idx < len(sentences):
        group: list[str] = []
        token_count = 0
        end_idx = start_idx

        # Grow the group until we hit the token target
        while end_idx < len(sentences):
            candidate = sentences[end_idx]
            new_count = token_count + _count_tokens(candidate) + 1  # +1 for space
            if new_count > CHUNK_TARGET_TOKENS and group:
                break
            group.append(candidate)
            token_count = new_count
            end_idx += 1

        chunk_text_str = " ".join(group)
        text_hash = compute_text_hash(chunk_text_str)
        chunk_index = len(chunks)
        chunk_id = build_chunk_id(source_type, source_id, chunk_index, text_hash)

        # Page range for this chunk
        pages_in_chunk = sentence_pages[start_idx:end_idx]
        page_start = min(pages_in_chunk) if pages_in_chunk else 0
        page_end = max(pages_in_chunk) if pages_in_chunk else 0

        metadata: ChunkMetadata = {
            "source_type": source_type,
            "source_id": source_id,
            "source_uri": source_uri,
            "title": title,
            "company": company,
            "location": location,
            "page_start": page_start,
            "page_end": page_end,
            "record_id": record_id,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "text_hash": text_hash,
            "ingest_version": INGEST_VERSION,
            "embedding_model": EMBEDDING_MODEL,
        }

        chunks.append((chunk_text_str, metadata))

        # Advance with overlap
        advance = max(1, len(group) - CHUNK_OVERLAP_SENTENCES)
        start_idx += advance

    return chunks


def _find_sentence_page(
    sentence: str,
    page_map: list[tuple[int, int, str]],
) -> int:
    """Return the 1-based page number where *sentence* most likely appears."""
    for page_num, _, page_text in page_map:
        # Use first 60 chars of sentence as a probe (robust to minor whitespace diff)
        probe = sentence[:60]
        if probe in page_text:
            return page_num
    # Fall back to first page if not found
    return page_map[0][0] if page_map else 1
