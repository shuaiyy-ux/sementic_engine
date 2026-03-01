# DEVELOPMENT.md

> Last updated: 2026-03-01 — reflects the Job RAG Chatbot rewrite.

---

## 1. Architecture Overview

The project follows a clean 4-layer architecture:

```
data sources
   ├─ 60 PDF job descriptions (data/pdf_sources/)
   └─ 5 000–100 000 LinkedIn job listings (data/linkedin_jobs_cleaned.sqlite)
          ↓
     ingest/          ← chunking, embedding, deduplication, Chroma upsert
          ↓
     storage/         ← Chroma vectorstore (persistent), manifest.sqlite
          ↓
     rag/             ← retriever, reranker, prompt builder, LLM generator
          ↓
     app/             ← Streamlit chat UI + citation cards
```

---

## 2. Key Decisions (locked — do not change without updating manifest version)

| Decision point | Chosen option | Rationale |
|---|---|---|
| **PDF extraction library** | `pdfplumber` | More accurate paragraph/table extraction vs `pypdf`; clean text output from structured govt JD PDFs |
| **Chunking strategy** | Sentence-level with 400-token window + 2-sentence overlap | Good citation granularity; sentence boundaries preserve semantics better than fixed char splits |
| **Vector store** | Chroma `PersistentClient` | Native incremental upsert; no manual serialization unlike FAISS |
| **Embedding model** | `all-MiniLM-L6-v2` (384-dim) | Already used in existing pipeline; fast inference; good quality for retrieval |
| **LLM** | `gpt-4o-mini` | Better instruction following than 3.5-turbo for citation format; lower cost than GPT-4o |
| **Advanced feature** | Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) | Cost-free (~10 ms/pair); proven on passage-retrieval benchmarks; easily demonstrable rank-change logs |
| **Follow-up handling** | Query rewriting via GPT-4o-mini | Simpler than maintaining a summary; works well within 8-turn window |
| **Deployment** | Streamlit Community Cloud | Free; one-click; no infra management |
| **OCR for scanned PDFs** | Skipped (not OCR) | All 60 PDFs have selectable text; OCR adds significant complexity for no benefit here |

---

## 3. Module Reference

### `app/config.py`
Central constants: all paths, model names, K values, temperature. **Import from here — never hardcode paths in other modules.**

### `ingest/build_index.py`
Orchestrator. Run via:
```bash
./run_ingest.sh                 # full index
./run_ingest.sh --skip-jobs     # PDFs only
./run_ingest.sh --job-limit 500 # smoke test
```

### `ingest/chunking.py`
Sentence-level chunking with `nltk.sent_tokenize` + `tiktoken` token counting.  
`CHUNK_TARGET_TOKENS = 400`, `CHUNK_OVERLAP_SENTENCES = 2`.  
Same (text, config) → same chunk_id set (deterministic).

### `ingest/metadata.py`
`ChunkMetadata` TypedDict, `build_chunk_id()`, `compute_text_hash()`.  
`chunk_id` format: `{source_type}_{source_id}_{chunk_index}_{hash_prefix_8chars}`

### `ingest/dedup.py`
Checks Chroma collection for existing `chunk_id`s — skips already-indexed chunks.

### `storage/manifest.sqlite`
Table `ingest_runs` records every build: time, version, counts, failed docs, duration.  
Inspect via:
```bash
sqlite3 storage/manifest.sqlite "SELECT * FROM ingest_runs ORDER BY run_id DESC LIMIT 5"
```

### `rag/retriever.py`
Wraps Chroma `collection.query()`. Returns top-N chunks, elapsed time.  
Supports `where` metadata filter (source_type, company, location).

### `rag/reranker.py`
`CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')`.  
Logs rank-before/rank-after to `storage/rerank_log.jsonl` for technical writeup.

### `rag/generator.py`
Full pipeline: rewrite query → retrieve 20 → rerank to 5 → build prompt → GPT-4o-mini → format citations.

### `app/streamlit_app.py`
Main entry point. Run with:
```bash
streamlit run app/streamlit_app.py
```

---

## 4. Data Flow (single query)

```
user input
   → [query rewrite?] if recent_turns, rewrite with GPT-4o-mini
   → embed with all-MiniLM-L6-v2
   → Chroma.query(n=20) + optional where-filter
   → CrossEncoder.predict(query, top-20 chunks) → sort → top-5
   → build_citations(top-5)
   → build_user_prompt(query, top-5 chunks, last-4 turns)
   → GPT-4o-mini (system + user prompt) → answer with [1][2] citations
   → format + render in Streamlit chat UI
```

---

## 5. Ingest Version & Rebuild Strategy

`INGEST_VERSION = "1.0.0"` in `app/config.py`.  
If you change the embedding model:
1. Bump `INGEST_VERSION` to `"2.0.0"`.
2. Delete `storage/vectorstore/` to force full rebuild.
3. Re-run `./run_ingest.sh`.

Changing only chunking config also requires a full rebuild (chunk_ids will differ).

---

## 6. Running Locally

```bash
# 1. Install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Set API key
cp .env.example .env && vi .env  # set OPENAI_API_KEY

# 3. Build index
./run_ingest.sh

# 4. Launch UI
streamlit run app/streamlit_app.py
```

---

## 7. File Structure

```
sementic_engine/
├── app/
│   ├── __init__.py
│   ├── streamlit_app.py      ← main Streamlit app
│   ├── ui_components.py      ← chat bubbles, citation cards
│   └── config.py             ← all constants & paths
├── ingest/
│   ├── __init__.py
│   ├── build_index.py        ← orchestrator (run this)
│   ├── ingest_pdf.py
│   ├── ingest_joblistings.py
│   ├── chunking.py
│   ├── metadata.py
│   └── dedup.py
├── rag/
│   ├── __init__.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── citations.py
│   ├── prompt.py
│   └── generator.py
├── storage/
│   ├── vectorstore/          ← Chroma persistent index (gitignored)
│   └── manifest.sqlite       ← ingest run history (gitignored)
├── data/
│   ├── pdf_sources/          ← 60 govt JD PDFs
│   └── linkedin_jobs_cleaned.sqlite  ← gitignored; download separately
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DEVELOPMENT.md        ← this file
│   └── TECHNICAL_WRITEUP.md
├── run_ingest.sh             ← convenience ingest runner
├── requirements.txt
├── .env.example
└── readme.md
```

---

## 8. Legacy Files (keep for reference, not part of production pipeline)

- `Preprocessing.ipynb` — original data cleaning notebook
- `semantic.ipynb` — original prototype RAG notebook
- `streamlit_app.py` (root) — original search-only Streamlit app
- `upload_db_to_hf.py` — HF Hub upload script
