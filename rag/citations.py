"""Citation builder: convert reranked chunks into structured citation objects."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Citation:
    """Structured reference to a retrieved chunk."""

    index: int           # 1-based citation number shown to user as [N]
    title: str
    source_type: str     # "pdf" | "job_listing"
    source_id: str       # PDF filename or job_id string
    source_uri: str      # Public URL or empty string
    page_start: int      # 0 = not applicable
    page_end: int
    record_id: str       # job listing record ID (empty for PDFs)
    preview_text: str    # First 300 chars of chunk text
    chunk_id: str
    company: str = ""
    location: str = ""


def build_citations(chunks: list[dict]) -> list[Citation]:
    """Create :class:`Citation` objects from *chunks*.

    Parameters
    ----------
    chunks:
        List of chunk dicts (each with ``"text"`` and ``"metadata"`` keys)
        ordered as they should appear in the citation list.

    Returns
    -------
    List of :class:`Citation` objects with 1-based ``index`` values.
    """
    citations: list[Citation] = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        text = chunk.get("text", "")
        preview = text[:300].strip()
        if len(text) > 300:
            preview += " …"

        citations.append(
            Citation(
                index=i,
                title=meta.get("title", "Unknown"),
                source_type=meta.get("source_type", ""),
                source_id=meta.get("source_id", ""),
                source_uri=meta.get("source_uri", ""),
                page_start=int(meta.get("page_start", 0)),
                page_end=int(meta.get("page_end", 0)),
                record_id=meta.get("record_id", ""),
                preview_text=preview,
                chunk_id=meta.get("chunk_id", ""),
                company=meta.get("company", ""),
                location=meta.get("location", ""),
            )
        )
    return citations


def format_citation_markers(answer: str, citations: list[Citation]) -> str:
    """Ensure [N] markers match the actual citations.

    If the LLM already inserted markers we trust them; otherwise we append
    a references section.  This function is defensive — it never removes
    existing markers.
    """
    # Check whether the answer already contains at least [1]
    if "[1]" in answer:
        return answer

    # Append a minimal references note
    if not citations:
        return answer

    refs = "\n\n**References**\n" + "\n".join(
        f"[{c.index}] {c.title} ({c.source_type})" for c in citations
    )
    return answer + refs
