import sqlite3
import tempfile
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="LinkedIn Job Semantic Search", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# Database download helper (for cloud deployment)
# ─────────────────────────────────────────────────────────────────────────────
# Primary: GitHub Release (simpler, no extra account needed)
# Fallback: Hugging Face Hub
DB_URLS = [
    "https://github.com/shuaiyy-ux/sementic_engine/releases/download/v1.0/linkedin_jobs_cleaned.sqlite",
    "https://huggingface.co/datasets/Fhujnfjfj/linkedin-jobs-sqlite/resolve/main/linkedin_jobs_cleaned.sqlite",
]
LOCAL_DB_NAME = "linkedin_jobs_cleaned.sqlite"


@st.cache_resource
def download_database():
    """Download the database from GitHub Releases or HF if not present locally."""
    import urllib.request
    
    # Check multiple possible locations
    candidates = [
        Path(__file__).parent / LOCAL_DB_NAME,
        Path(__file__).parent.parent / LOCAL_DB_NAME,
        Path(LOCAL_DB_NAME),
        Path("/tmp") / LOCAL_DB_NAME,  # For Streamlit Cloud
    ]
    
    for cand in candidates:
        if cand.exists() and cand.stat().st_size > 1000:
            return str(cand.resolve())
    
    # Download to /tmp for cloud environments
    target = Path("/tmp") / LOCAL_DB_NAME
    if not target.exists():
        for url in DB_URLS:
            st.info(f"📥 Downloading database... This may take a minute.")
            try:
                urllib.request.urlretrieve(url, str(target))
                if target.exists() and target.stat().st_size > 1000:
                    st.success("✅ Database downloaded successfully!")
                    return str(target)
            except Exception as e:
                st.warning(f"Failed from {url.split('/')[2]}: {e}")
                continue
        st.error("❌ Could not download database from any source.")
        return None
    return str(target)


# ─────────────────────────────────────────────────────────────────────────────
# Caching functions
# ─────────────────────────────────────────────────────────────────────────────
def read_tables_from_db(db_path: str):
    conn = sqlite3.connect(db_path)
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';", conn
    )
    table_names = tables["name"].tolist()
    return conn, table_names


@st.cache_data
def load_table(db_path: str, table_name: str, limit: int = 10000):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f'SELECT * FROM "{table_name}" LIMIT {limit};', conn)
    conn.close()
    return df


@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def compute_embeddings(texts: tuple):
    model = load_embedding_model()
    return model.encode(list(texts), show_progress_bar=False)


# ─────────────────────────────────────────────────────────────────────────────
# UI Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("🔍 LinkedIn Job Semantic Search Engine")

st.markdown(
    """
    **Semantic search** powered by `sentence-transformers` (all-MiniLM-L6-v2).  
    Upload your `*.sqlite` file or use the bundled database. Enter a query to find
    the most relevant jobs by meaning — not just keywords!
    """
)

# ─────────────────────────────────────────────────────────────────────────────
# Database loading
# ─────────────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a SQLite file", type=["sqlite", "db"], accept_multiple_files=False)

db_path = None
table_names = []

if uploaded is not None:
    t = tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite")
    t.write(uploaded.getvalue())
    t.flush()
    t.close()
    db_path = t.name
    _, table_names = read_tables_from_db(db_path)
    st.success(f"✅ Loaded {uploaded.name}")
else:
    # Try to find or download database
    db_path = download_database()
    if db_path:
        _, table_names = read_tables_from_db(db_path)
        st.info(f"📂 Using database: `{Path(db_path).name}`")

if not table_names or db_path is None:
    st.warning("No tables found. Upload a sqlite file containing `job_listings_cleaned`.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: table selection & settings
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
selected_table = st.sidebar.selectbox("Select table", table_names, index=table_names.index("job_listings_cleaned") if "job_listings_cleaned" in table_names else 0)
max_rows = st.sidebar.slider("Max rows to load", 100, 10000, 2000, step=100)
top_k = st.sidebar.slider("Top K results", 5, 50, 10)

df = load_table(db_path, selected_table, limit=max_rows)

# ─────────────────────────────────────────────────────────────────────────────
# Main tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Semantic Search", "📋 Browse Jobs", "📊 Statistics"])

with tab1:
    st.subheader("Semantic Job Search")
    query = st.text_input("Enter your job search query", placeholder="e.g., machine learning engineer with Python experience")

    if query and selected_table == "job_listings_cleaned":
        with st.spinner("Computing embeddings..."):
            # Prepare texts
            descriptions = df["description"].fillna("").tolist()
            titles = df["title"].fillna("").tolist()
            combined = [f"{t}. {d[:500]}" for t, d in zip(titles, descriptions)]

            # Compute embeddings
            doc_embeddings = compute_embeddings(tuple(combined))
            model = load_embedding_model()
            query_embedding = model.encode([query], show_progress_bar=False)

            # Compute cosine similarity
            similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(similarities)[::-1][:top_k]

        st.success(f"Found top {top_k} most relevant jobs")

        for rank, idx in enumerate(top_indices, 1):
            score = similarities[idx]
            row = df.iloc[idx]
            with st.expander(f"**#{rank}** | Score: {score:.4f} | {row.get('title', 'N/A')}", expanded=(rank <= 3)):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Relevance", f"{score:.2%}")
                    if "company_id" in row and pd.notna(row["company_id"]):
                        st.write(f"**Company ID:** {int(row['company_id'])}")
                    if "job_id" in row:
                        st.write(f"**Job ID:** {row['job_id']}")
                with col2:
                    st.markdown(f"**Title:** {row.get('title', 'N/A')}")
                    desc = row.get("description", "")
                    if len(desc) > 800:
                        st.markdown(f"{desc[:800]}...")
                        with st.popover("Read full description"):
                            st.markdown(desc)
                    else:
                        st.markdown(desc if desc else "_No description_")

    elif query:
        st.warning("Semantic search only works on `job_listings_cleaned` table.")

with tab2:
    st.subheader(f"Browse: {selected_table}")

    # Keyword filter
    keyword = st.text_input("Filter by keyword (title/description)", key="browse_filter")
    filtered_df = df
    if keyword and selected_table == "job_listings_cleaned":
        mask = (
            df["title"].fillna("").str.contains(keyword, case=False, na=False)
            | df["description"].fillna("").str.contains(keyword, case=False, na=False)
        )
        filtered_df = df[mask]
        st.write(f"Showing {len(filtered_df)} of {len(df)} rows")

    st.dataframe(filtered_df.head(200), width="stretch")

    # Job detail viewer
    if selected_table == "job_listings_cleaned" and len(filtered_df) > 0:
        st.markdown("---")
        st.subheader("📄 Job Detail Viewer")
        job_idx = st.number_input("Enter row index to view details", min_value=0, max_value=len(filtered_df)-1, value=0)
        job = filtered_df.iloc[job_idx]
        st.markdown(f"### {job.get('title', 'N/A')}")
        st.markdown(f"**Job ID:** {job.get('job_id', 'N/A')} | **Company ID:** {job.get('company_id', 'N/A')}")
        st.markdown("#### Description")
        st.markdown(job.get("description", "_No description available_"))

with tab3:
    st.subheader("📊 Dataset Statistics")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Jobs", len(df))
    with col2:
        if "title" in df.columns:
            st.metric("Unique Titles", df["title"].nunique())
    with col3:
        if "company_id" in df.columns:
            st.metric("Unique Companies", df["company_id"].nunique())

    if selected_table == "job_listings_cleaned":
        st.markdown("---")
        st.subheader("Top Job Titles")
        if "clean_title" in df.columns:
            title_counts = df["clean_title"].value_counts().head(20)
        else:
            title_counts = df["title"].value_counts().head(20)
        st.bar_chart(title_counts)

        st.markdown("---")
        st.subheader("Description Length Distribution")
        if "description" in df.columns:
            df["desc_len"] = df["description"].fillna("").str.len()
            # Create histogram using numpy bins and bar_chart
            counts, bins = np.histogram(df["desc_len"], bins=30)
            hist_df = pd.DataFrame({"Characters": bins[:-1], "Count": counts})
            st.bar_chart(hist_df, x="Characters", y="Count")

    st.markdown("---")
    st.subheader("Column Info")
    st.write(list(df.columns))
    dtype_df = df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Type"})
    dtype_df["Type"] = dtype_df["Type"].astype(str)
    st.dataframe(dtype_df)
