# Job RAG Chatbot

A conversational retrieval-augmented generation (RAG) system over job descriptions.  
Ask natural-language questions and get cited, LLM-generated answers backed by 60 PDF job descriptions and thousands of LinkedIn job listings.

**Tech stack:** Python · Chroma · Sentence Transformers · GPT-4o-mini · Streamlit · SQLite

---

## Features

| Feature | Detail |
|---|---|
| **Multi-source retrieval** | 60 government/municipal PDF JDs + LinkedIn job listings from SQLite |
| **Persistent vector index** | Chroma with incremental upsert — no re-embedding on re-run |
| **Sentence-level chunking** | ~400-token windows with 2-sentence overlap, page-number anchoring |
| **Cross-encoder reranking** | `ms-marco-MiniLM-L-6-v2`: retrieve top-20 → rerank → pass top-5 to LLM |
| **Structured citations** | `[1][2]` inline markers + expandable reference cards in UI |
| **Multi-turn memory** | Last 8 turns kept in session; follow-up queries auto-rewritten |
| **Streamlit Chat UI** | Chat bubbles, citation cards, latency metrics, sidebar filters |

---

## Quick start

### 1. Clone & install

```bash
git clone <your-repo-url>
cd sementic_engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 3. Add the SQLite database

Download `linkedin_jobs_cleaned.sqlite` and place it at:

```
data/linkedin_jobs_cleaned.sqlite
```

Download link: https://github.com/peterhan0504-dev/Semantic-Search-Engine/releases/tag/v1.0-data

### 4. Build the vector index

```bash
# Full index (all 60 PDFs + all job listings)
python -m ingest.build_index

# Smoke-test with first 500 job listings only
python -m ingest.build_index --job-limit 500

# Skip job listings (PDFs only)
python -m ingest.build_index --skip-jobs
```

The first run downloads the `all-MiniLM-L6-v2` and `ms-marco-MiniLM-L-6-v2` model weights (~100 MB total).  
Subsequent runs skip already-indexed chunks (deduplication via `chunk_id`).

### 5. Launch the app

```bash
streamlit run app/streamlit_app.py
```

---

## Project structure

```
sementic_engine/
├── app/
│   ├── streamlit_app.py     ← Main Streamlit chat application
│   ├── ui_components.py     ← Chat bubbles, citation cards, sidebar
│   └── config.py            ← All constants and path definitions
├── ingest/
│   ├── build_index.py       ← Orchestrator (run this to build the index)
│   ├── ingest_pdf.py        ← PDF extraction via pdfplumber
│   ├── ingest_joblistings.py← SQLite → chunks
│   ├── chunking.py          ← Sentence-level chunking with token windows
│   ├── metadata.py          ← ChunkMetadata schema and chunk_id builder
│   └── dedup.py             ← chunk_id-based deduplication
├── rag/
│   ├── retriever.py         ← Chroma vector retrieval
│   ├── reranker.py          ← Cross-encoder reranking (top-20 → top-5)
│   ├── citations.py         ← Citation dataclass and formatter
│   ├── prompt.py            ← System prompt, user prompt, rewrite prompt
│   └── generator.py         ← Full RAG pipeline (retrieve → rerank → generate)
├── storage/
│   ├── vectorstore/         ← Chroma persistent index (gitignored)
│   └── manifest.sqlite      ← Ingest run history (gitignored)
├── data/
│   ├── pdf_sources/         ← 60 government job description PDFs
│   └── linkedin_jobs_cleaned.sqlite  ← (gitignored, download separately)
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DEVELOPMENT.md
│   └── TECHNICAL_WRITEUP.md
├── .env.example
└── requirements.txt
```

---

## Deploying to Streamlit Community Cloud

1. Push the repo to GitHub (the `storage/vectorstore/` and `.env` are gitignored).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your repo.
3. Set **Main file path** to `app/streamlit_app.py`.
4. Under **Secrets**, add:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
5. **Important:** The Chroma index must be available at deploy time. Options:
   - Check in a pre-built index (remove `storage/vectorstore/` from `.gitignore` and commit it for small indexes).
   - Or add a startup script that calls `python -m ingest.build_index` before the app boots.

---

## Data sources & copyright

| Source | Licence / Usage |
|---|---|
| LinkedIn job listings | [datastax/linkedin_job_listings](https://huggingface.co/datasets/datastax/linkedin_job_listings) — public HF dataset, research use |
| Government PDF JDs | Public domain city/municipal job classification documents — no copyright restrictions |

---

## Key design decisions

- **Chroma over FAISS** — `PersistentClient` supports incremental upsert natively.
- **Sentence-level chunking** — better citation granularity vs fixed-length windows.
- **Cross-encoder reranking** — ~10 ms per pair, no extra API cost, proven on passage retrieval benchmarks.
- **GPT-4o-mini** — better instruction following for citation format at significantly lower cost than GPT-4o.
- **Query rewriting** — LLM rewrites follow-ups into standalone queries before retrieval; handles pronouns and references to prior turns.

