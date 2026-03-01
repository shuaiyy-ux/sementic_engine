"""Reusable Streamlit UI components for the Job RAG Chatbot."""

from __future__ import annotations

import streamlit as st

from rag.citations import Citation


# ── Badge helpers ─────────────────────────────────────────────────────────────

_BADGE_COLORS = {
    "pdf": "#2196F3",
    "job_listing": "#4CAF50",
}


def _source_badge(source_type: str) -> str:
    color = _BADGE_COLORS.get(source_type, "#9E9E9E")
    label = "PDF" if source_type == "pdf" else "Job Listing"
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.75rem;font-weight:600;">{label}</span>'
    )


# ── Citation card ─────────────────────────────────────────────────────────────

def render_citation_card(citation: Citation) -> None:
    """Render a single citation as a collapsible expander."""
    label_parts = [f"[{citation.index}] {citation.title}"]
    if citation.source_type == "pdf" and citation.page_start:
        label_parts.append(f"— p.{citation.page_start}")
        if citation.page_end and citation.page_end != citation.page_start:
            label_parts[-1] = f"— pp.{citation.page_start}–{citation.page_end}"
    if citation.company:
        label_parts.append(f"| {citation.company}")

    with st.expander(" ".join(label_parts), expanded=False):
        # Source badge + meta
        st.markdown(_source_badge(citation.source_type), unsafe_allow_html=True)
        st.write("")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Source ID:** `{citation.source_id}`")
            if citation.source_type == "pdf":
                page_str = (
                    f"{citation.page_start}"
                    if citation.page_start == citation.page_end
                    else f"{citation.page_start}–{citation.page_end}"
                )
                st.markdown(f"**Pages:** {page_str or 'N/A'}")
            else:
                st.markdown(f"**Record ID:** {citation.record_id or 'N/A'}")
        with col2:
            if citation.company:
                st.markdown(f"**Company:** {citation.company}")
            if citation.location:
                st.markdown(f"**Location:** {citation.location}")
            if citation.source_uri:
                st.markdown(f"**Source:** [{citation.source_uri[:50]}]({citation.source_uri})")

        st.divider()
        st.markdown("**Chunk preview:**")
        st.markdown(
            f'<div style="background:#f5f5f5;padding:10px;border-radius:6px;'
            f'font-size:0.85rem;white-space:pre-wrap;">{citation.preview_text}</div>',
            unsafe_allow_html=True,
        )


def render_citation_cards(citations: list[Citation]) -> None:
    """Render all citation cards in a labelled section."""
    if not citations:
        return
    st.markdown("#### References")
    for citation in citations:
        render_citation_card(citation)


# ── Chat message ──────────────────────────────────────────────────────────────

def render_chat_message(
    role: str,
    content: str,
    citations: list[Citation] | None = None,
    timing: dict | None = None,
) -> None:
    """Render a single chat turn with optional citations and timing info."""
    with st.chat_message(role):
        st.markdown(content)

        if role == "assistant":
            if citations:
                render_citation_cards(citations)
            if timing:
                with st.expander("⏱ Latency", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Retrieve", f"{timing.get('retrieve_s', 0):.3f}s")
                    col2.metric("Rerank", f"{timing.get('rerank_s', 0):.3f}s")
                    col3.metric("Generate", f"{timing.get('generate_s', 0):.2f}s")


# ── Sidebar filters ───────────────────────────────────────────────────────────

def render_sidebar_filters() -> dict | None:
    """Render filter controls in the sidebar and return a Chroma where-filter.

    Returns ``None`` if no filters are active.
    """
    st.sidebar.header("Filters")

    source_type = st.sidebar.selectbox(
        "Source type",
        options=["All", "pdf", "job_listing"],
        index=0,
    )
    company = st.sidebar.text_input("Company (job listings only)", value="")
    location = st.sidebar.text_input("Location (job listings only)", value="")

    filters: dict[str, str] = {}
    if source_type != "All":
        filters["source_type"] = source_type
    if company.strip():
        filters["company"] = company.strip()
    if location.strip():
        filters["location"] = location.strip()

    return filters if filters else None


# ── Manifest stats sidebar ────────────────────────────────────────────────────

def render_manifest_stats() -> None:
    """Display index stats from the manifest SQLite in the sidebar.

    Aggregates across all ingest runs so that a PDF-only run followed by a
    job-only run still shows correct totals for both source types.
    """
    import sqlite3
    from app.config import MANIFEST_DB

    if not MANIFEST_DB.exists():
        st.sidebar.info("Index not yet built. Run `./run_ingest.sh`.")
        return

    try:
        conn = sqlite3.connect(MANIFEST_DB)
        # Latest run timestamp and current total chunks
        last_row = conn.execute(
            "SELECT ingest_time, total_chunks FROM ingest_runs ORDER BY run_id DESC LIMIT 1"
        ).fetchone()
        # Aggregate: max pdf/job counts seen across all runs (handles split runs)
        agg_row = conn.execute(
            "SELECT MAX(pdf_count), MAX(job_count) FROM ingest_runs"
        ).fetchone()
        conn.close()
    except Exception:
        return

    if last_row and agg_row:
        st.sidebar.header("Index Stats")
        st.sidebar.markdown(f"**Last indexed:** {last_row[0][:19].replace('T', ' ')} UTC")
        st.sidebar.markdown(f"**Total chunks:** {last_row[1]:,}")
        st.sidebar.markdown(f"**PDFs:** {agg_row[0] or 0} | **Jobs:** {(agg_row[1] or 0):,}")
