# DEVELOPMENT.md

## 1. File Functionality Overview

- `Preprocessing.ipynb`: Cleans and preprocesses LinkedIn job listings data from the Hugging Face dataset (`datastax/linkedin_job_listings`). Outputs a cleaned SQLite database (`linkedin_jobs_cleaned.sqlite`).
- `semantic.ipynb`: Loads the cleaned SQLite database, generates sentence embeddings using `sentence-transformers/all-MiniLM-L6-v2`, and performs semantic search (cosine similarity) and retrieval-augmented generation (RAG) with LLM QA.
- `streamlit_app.py`: Provides a Streamlit web interface for semantic search over job listings. Handles database loading, user queries, and result display.
- `upload_db_to_hf.py`: Script to upload the cleaned SQLite database to the Hugging Face Hub for sharing and reproducibility.
- `requirements.txt`: Lists all Python dependencies required for the project.
- `ARCHITECTURE.md`: Documents the system architecture, data flow, and design notes.
- `TEAM_CONTRIBUTIONS.md`: Records individual contributions to the project.
- `readme.md`: Project overview, usage instructions, and file descriptions.
- `linkedin_jobs_sample.sqlite`: Sample SQLite database for demo purposes.

## 2. Design Principles

- **Reproducibility**: All data processing and model steps are documented in notebooks/scripts. Database and code are versioned for reproducibility.
- **Modularity**: Data processing, embedding, retrieval, and interface are separated into distinct files/modules.
- **Transparency**: Notebooks and markdown files document each step, rationale, and design decision.
- **Scalability**: Supports large datasets (100k+ job listings) and can be extended to new domains or models.
- **Accessibility**: Provides both CLI and web (Streamlit) interfaces for diverse user needs.

## 3. Data Pipeline

1. **Data Collection & Preprocessing**
   - Source: Hugging Face dataset (`datastax/linkedin_job_listings`).
   - Cleaned and filtered in `Preprocessing.ipynb`.
   - Output: `linkedin_jobs_cleaned.sqlite` (table: `job_listings_cleaned`).

2. **Embedding Generation**
   - Performed in `semantic.ipynb` using `sentence-transformers/all-MiniLM-L6-v2`.
   - Embeddings are computed for job descriptions and used for similarity search.

3. **Semantic Search & Retrieval**
   - User queries are embedded and compared to job embeddings using cosine similarity.
   - Top-K relevant jobs are retrieved.

4. **LLM QA (RAG)**
   - Retrieved job descriptions are concatenated as context.
   - Context and user query are sent to an LLM (e.g., OpenAI GPT-3.5) for answer generation.

5. **Deployment & Sharing**
   - Web interface via Streamlit (`streamlit_app.py`).
   - Database upload to Hugging Face Hub (`upload_db_to_hf.py`).

## 4. File Structure

```
sementic_engine/
├── ARCHITECTURE.md
├── DEVELOPMENT.md
├── Preprocessing.ipynb
├── TEAM_CONTRIBUTIONS.md
├── linkedin_jobs_sample.sqlite
├── readme.md
├── requirements.txt
├── semantic.ipynb
├── streamlit_app.py
├── upload_db_to_hf.py
```

- Notebooks for data and model steps
- Scripts for deployment and sharing
- Markdown files for documentation
- SQLite DBs for data storage
