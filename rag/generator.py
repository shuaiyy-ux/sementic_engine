"""Main RAG generator: ties together retrieval, reranking and LLM generation."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from openai import OpenAI

from app.config import (
    FINAL_K,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
    RETRIEVE_N,
    STORAGE_DIR,
)
from rag.citations import Citation, build_citations, format_citation_markers
from rag.prompt import SYSTEM_PROMPT, build_rewrite_prompt, build_user_prompt
from rag.reranker import rerank
from rag.retriever import retrieve

# Optional rerank log file
_RERANK_LOG = STORAGE_DIR / "rerank_log.jsonl"


def _get_openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file or environment."
        )
    return OpenAI(api_key=OPENAI_API_KEY)


def _rewrite_query(
    raw_query: str,
    recent_turns: list[dict],
    client: OpenAI,
) -> str:
    """Use GPT-4o-mini to rewrite a follow-up query into a standalone query."""
    if not recent_turns:
        return raw_query
    try:
        prompt = build_rewrite_prompt(raw_query, recent_turns)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
        )
        rewritten = resp.choices[0].message.content.strip()
        if rewritten:
            print(f"[Generator] Query rewritten: {rewritten!r}", file=sys.stderr)
            return rewritten
    except Exception as exc:
        print(f"[Generator] Rewrite failed ({exc}), using raw query", file=sys.stderr)
    return raw_query


def _append_rerank_log(entry: dict) -> None:
    """Append a JSON line to the rerank log file."""
    try:
        with _RERANK_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # log write failure is non-fatal


def generate(
    query: str,
    recent_turns: list[dict] | None = None,
    where_filter: dict | None = None,
) -> tuple[str, list[Citation], dict]:
    """Run the full RAG pipeline for a single user turn.

    Parameters
    ----------
    query:
        Raw user question.
    recent_turns:
        Recent ``{role, content}`` dicts from session memory (for follow-up
        query rewriting).
    where_filter:
        Optional Chroma metadata filter (e.g. ``{"source_type": "pdf"}``).

    Returns
    -------
    ``(answer, citations, timing)``

    ``timing`` is a dict with ``retrieve_s``, ``rerank_s``, ``generate_s``
    floats for observability.
    """
    recent_turns = recent_turns or []
    client = _get_openai_client()

    # ── Query rewrite (follow-up handling) ────────────────────────────────────
    effective_query = _rewrite_query(query, recent_turns, client)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    candidates, retrieve_s = retrieve(
        effective_query,
        n_results=RETRIEVE_N,
        where=where_filter,
    )
    print(
        f"[Generator] Retrieved {len(candidates)} candidates in {retrieve_s:.3f}s",
        file=sys.stderr,
    )

    if not candidates:
        no_evidence = (
            "I could not find relevant information in the knowledge base for your query. "
            "Try rephrasing or broadening your search terms."
        )
        return no_evidence, [], {"retrieve_s": retrieve_s, "rerank_s": 0.0, "generate_s": 0.0}

    # ── Reranking ─────────────────────────────────────────────────────────────
    top_chunks, rerank_s, rank_log = rerank(effective_query, candidates, top_k=FINAL_K)

    # Persist rank log
    _append_rerank_log(
        {
            "query": effective_query,
            "original_query": query,
            "rank_log": rank_log,
        }
    )

    # ── Build citations ───────────────────────────────────────────────────────
    citations = build_citations(top_chunks)

    # ── Conversation context string ────────────────────────────────────────────
    conv_context = ""
    if recent_turns:
        lines = []
        for t in recent_turns[-4:]:
            role = t.get("role", "user").capitalize()
            content = (t.get("content") or "")[:200]
            lines.append(f"{role}: {content}")
        conv_context = "\n".join(lines)

    # ── LLM generation ────────────────────────────────────────────────────────
    user_prompt = build_user_prompt(
        query=query,  # show the original query to the LLM, not the rewritten one
        chunks=top_chunks,
        conversation_context=conv_context,
    )

    t_gen = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as exc:
        answer = f"[Error calling LLM: {exc}]"
        print(f"[Generator] LLM error: {exc}", file=sys.stderr)
    generate_s = time.monotonic() - t_gen

    answer = format_citation_markers(answer, citations)

    timing = {
        "retrieve_s": round(retrieve_s, 3),
        "rerank_s": round(rerank_s, 3),
        "generate_s": round(generate_s, 3),
    }
    print(
        f"[Generator] generate_s={timing['generate_s']:.2f}  "
        f"retrieve_s={timing['retrieve_s']:.3f}  rerank_s={timing['rerank_s']:.3f}",
        file=sys.stderr,
    )
    return answer, citations, timing
