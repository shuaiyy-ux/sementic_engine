"""Job RAG Chatbot — main Streamlit application.

Run locally:
    streamlit run app/streamlit_app.py

The app expects the Chroma vector index to be built first:
    python -m ingest.build_index

Environment variable required:
    OPENAI_API_KEY  — set in .env (local) or Streamlit Cloud secrets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ── Ensure project root is importable when launched from app/ ─────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Job RAG Chatbot",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.config import MEMORY_WINDOW, OPENAI_API_KEY
from app.ui_components import (
    render_chat_message,
    render_manifest_stats,
    render_sidebar_filters,
)
from rag.generator import generate


# ── Session state helpers ─────────────────────────────────────────────────────

def _init_session() -> None:
    """Initialise session state keys on first load."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # list of {role, content, citations, timing}
    if "show_debug" not in st.session_state:
        st.session_state["show_debug"] = False


def _get_recent_turns() -> list[dict]:
    """Return last MEMORY_WINDOW messages formatted for the LLM."""
    msgs = st.session_state.get("messages", [])
    window = msgs[-MEMORY_WINDOW:]
    return [{"role": m["role"], "content": m.get("content", "")} for m in window]


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> dict | None:
    st.sidebar.title("💼 Job RAG Chatbot")
    st.sidebar.markdown(
        "Ask questions about job roles, required skills, responsibilities, and more. "
        "The chatbot searches both PDF job descriptions and LinkedIn job listings."
    )
    st.sidebar.divider()

    where_filter = render_sidebar_filters()

    st.sidebar.divider()
    render_manifest_stats()

    st.sidebar.divider()
    if st.sidebar.button("🗑️ Clear conversation"):
        st.session_state["messages"] = []
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption("Powered by GPT-4o-mini · Chroma · all-MiniLM-L6-v2")

    return where_filter


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_session()

    where_filter = _render_sidebar()

    st.title("💼 Job RAG Chatbot")
    st.caption(
        "Ask me about job roles, skills, responsibilities, salary, and career paths. "
        "I'll search the knowledge base and cite my sources."
    )

    # ── API key warning ───────────────────────────────────────────────────────
    if not OPENAI_API_KEY:
        st.warning(
            "⚠️ **OPENAI_API_KEY is not configured.** "
            "Add it to a `.env` file in the project root, or set it as a "
            "Streamlit Cloud secret, to enable answer generation.",
            icon="⚠️",
        )

    # ── Render existing conversation ──────────────────────────────────────────
    for msg in st.session_state["messages"]:
        render_chat_message(
            role=msg["role"],
            content=msg["content"],
            citations=msg.get("citations"),
            timing=msg.get("timing"),
        )

    # ── Chat input ────────────────────────────────────────────────────────────
    user_input: str | None = st.chat_input(
        "Ask a question about jobs, skills or roles…",
        disabled=not OPENAI_API_KEY,
    )

    if user_input:
        user_input = user_input.strip()
        if not user_input:
            st.stop()

        # Append and immediately render the user message
        st.session_state["messages"].append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # ── Generate response ──────────────────────────────────────────────────
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base…"):
                try:
                    answer, citations, timing = generate(
                        query=user_input,
                        recent_turns=_get_recent_turns(),
                        where_filter=where_filter or None,
                    )
                except Exception as exc:
                    answer = f"An error occurred: {exc}"
                    citations = []
                    timing = {}

            st.markdown(answer)

            from app.ui_components import render_citation_cards
            render_citation_cards(citations)

            if timing:
                with st.expander("⏱ Latency", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Retrieve", f"{timing.get('retrieve_s', 0):.3f}s")
                    col2.metric("Rerank", f"{timing.get('rerank_s', 0):.3f}s")
                    col3.metric("Generate", f"{timing.get('generate_s', 0):.2f}s")

        # Save assistant turn to memory
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": answer,
                "citations": citations,
                "timing": timing,
            }
        )

        # Trim memory to MEMORY_WINDOW turns
        if len(st.session_state["messages"]) > MEMORY_WINDOW * 2:
            st.session_state["messages"] = st.session_state["messages"][-(MEMORY_WINDOW * 2):]


if __name__ == "__main__":
    main()
else:
    # Streamlit runs the module, so call main() unconditionally
    main()
