"""Prompt builder for the RAG generation step."""

from __future__ import annotations

from rag.citations import Citation

SYSTEM_PROMPT = """\
You are a helpful job search assistant with access to a knowledge base of job \
descriptions and job postings. Use the provided context chunks to answer the \
user's question accurately and specifically.

Guidelines:
- Answer in clear bullet points or numbered steps where appropriate.
- Cite your sources inline using [N] notation matching the context numbers below.
- If the provided context does not contain enough information to answer the question, \
explicitly say "The available evidence is insufficient to answer this fully." and \
suggest what additional information the user might search for.
- Do NOT make up job requirements or company details not present in the context.
- Keep your answer focused and concise (under 400 words unless the question requires more).
"""


def build_user_prompt(
    query: str,
    chunks: list[dict],
    conversation_context: str = "",
) -> str:
    """Build the user-turn prompt from the query, retrieved chunks and memory.

    Parameters
    ----------
    query:
        The (possibly rewritten) user question.
    chunks:
        Reranked chunks to use as context, each with ``"text"`` key.
    conversation_context:
        Short summary or recent turns from memory (prepended to the context).

    Returns
    -------
    A formatted string to be sent as the ``user`` message.
    """
    parts: list[str] = []

    if conversation_context:
        parts.append(f"**Conversation context:**\n{conversation_context}\n")

    parts.append("**Retrieved context:**\n")
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "").strip()
        meta = chunk.get("metadata", {})
        title = meta.get("title", "Unknown")
        source_type = meta.get("source_type", "")
        parts.append(f"[{i}] ({source_type}) {title}:\n{text}\n")

    parts.append(f"**Question:** {query}")

    return "\n".join(parts)


def build_rewrite_prompt(query: str, recent_turns: list[dict]) -> str:
    """Build the prompt that asks the LLM to rewrite a follow-up query.

    Parameters
    ----------
    query:
        The raw follow-up query from the user.
    recent_turns:
        Last few conversation messages, each a dict with ``role`` and
        ``content`` keys.

    Returns
    -------
    A compact rewrite instruction for a fast LLM call.
    """
    history_lines = []
    for turn in recent_turns[-4:]:  # at most 4 turns for context
        role = turn.get("role", "user")
        content = (turn.get("content", "") or "")[:300]
        history_lines.append(f"{role.capitalize()}: {content}")

    history_str = "\n".join(history_lines)

    return (
        f"Given the following conversation history:\n{history_str}\n\n"
        f"Rewrite the follow-up question below as a single, fully self-contained "
        f"search query that can be understood without reading the conversation "
        f"history. Output ONLY the rewritten query, no explanation.\n\n"
        f"Follow-up question: {query}"
    )
