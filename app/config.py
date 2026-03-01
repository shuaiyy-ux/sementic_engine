"""Central configuration for the Job RAG Chatbot.

All paths are resolved relative to the project root (parent of this file's
parent directory), so the app works correctly whether launched from the repo
root or from the ``app/`` sub-directory.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Project root ──────────────────────────────────────────────────────────────
# app/config.py  →  parent = app/  →  parent.parent = semantic_engine/
ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root (no-op if the file is missing)
load_dotenv(ROOT / ".env")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdf_sources"
SQLITE_DB = DATA_DIR / "linkedin_jobs_cleaned.sqlite"

STORAGE_DIR = ROOT / "storage"
VECTORSTORE_DIR = STORAGE_DIR / "vectorstore"
MANIFEST_DB = STORAGE_DIR / "manifest.sqlite"

# Ensure critical directories exist at import time
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ── Chroma ────────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "job_rag"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_TARGET_TOKENS = 400   # target chunk size in tokens
CHUNK_OVERLAP_SENTENCES = 2  # number of sentences to overlap between chunks

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVE_N = 20   # number of candidates to fetch from Chroma before reranking
FINAL_K = 5       # number of chunks sent to the LLM after reranking

# ── Reranker ──────────────────────────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 800

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

# ── Conversation ──────────────────────────────────────────────────────────────
MEMORY_WINDOW = 8  # number of recent turns to keep in session memory

# ── Ingest versioning ─────────────────────────────────────────────────────────
INGEST_VERSION = "1.0.0"
INGEST_BATCH_SIZE = 256  # chunk upsert batch size
