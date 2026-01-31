import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="LinkedIn Job Explorer", layout="wide")


def read_tables_from_db(db_path: str):
    conn = sqlite3.connect(db_path)
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';", conn)
    table_names = tables['name'].tolist()
    return conn, table_names


@st.cache_data
def load_table(conn, table_name: str, limit: int = 1000):
    df = pd.read_sql_query(f'SELECT * FROM "{table_name}" LIMIT {limit};', conn)
    return df


st.title("LinkedIn Job Explorer")

st.markdown(
    """
    Upload a `*.sqlite` file exported from the project (or drop your local copy).
    The app will list available tables and let you preview and search the
    `job_listings_cleaned` table by title/description.
    """
)

uploaded = st.file_uploader("Upload a SQLite file", type=["sqlite", "db"], accept_multiple_files=False)

default_db = Path("linkedin_jobs_cleaned.sqlite")

conn = None
table_names = []

if uploaded is not None:
    # Save to temp file
    t = tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite")
    t.write(uploaded.getvalue())
    t.flush()
    t.close()
    conn, table_names = read_tables_from_db(t.name)
    st.success(f"Loaded {uploaded.name}")
else:
    # try repo-local DB
    repo_db = Path(__file__).parent / "linkedin_jobs_cleaned.sqlite"
    if repo_db.exists():
        conn, table_names = read_tables_from_db(str(repo_db))
        st.info(f"Loaded bundled DB: {repo_db.name}")
    elif default_db.exists():
        conn, table_names = read_tables_from_db(str(default_db))
        st.info(f"Loaded local DB: {default_db}")

if not table_names:
    st.warning("No tables found. Upload a sqlite file containing `job_listings_cleaned`.")
    st.stop()

st.sidebar.header("Tables")
selected_table = st.sidebar.selectbox("Select table", table_names)

df = load_table(conn, selected_table, limit=5000)

st.subheader(f"Preview: {selected_table}")
st.dataframe(df.head(100))

if selected_table == "job_listings_cleaned":
    st.subheader("Search Jobs")
    q = st.text_input("Search query (title or description)")
    min_rows = st.slider("Minimum rows to show", 1, 200, 20)

    if q:
        mask = df['title'].fillna("").str.contains(q, case=False, na=False) | df['description'].fillna("").str.contains(q, case=False, na=False)
        results = df[mask].head(min_rows)
        st.write(f"Found {mask.sum()} matching rows")
        st.dataframe(results)

    st.markdown("---")
    st.write("Columns:")
    st.write(list(df.columns))

conn.close()
