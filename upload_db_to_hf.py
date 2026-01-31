#!/usr/bin/env python3
"""
Upload the SQLite database to Hugging Face Hub.

Usage:
    1. Install huggingface_hub: pip install huggingface_hub
    2. Login: huggingface-cli login
    3. Run: python upload_db_to_hf.py

This will create/update the dataset at:
    https://huggingface.co/datasets/shuaiyy-ux/linkedin-jobs-sqlite
"""

from huggingface_hub import HfApi, create_repo
from pathlib import Path

REPO_ID = "Fhujnfjfj/linkedin-jobs-sqlite"
DB_FILE = "linkedin_jobs_cleaned.sqlite"

def main():
    api = HfApi()
    
    # Find the database file
    db_path = None
    candidates = [
        Path(__file__).parent / DB_FILE,
        Path(__file__).parent.parent / DB_FILE,
        Path.home() / "Downloads" / DB_FILE,
    ]
    for cand in candidates:
        if cand.exists():
            db_path = cand
            break
    
    if not db_path:
        print(f"❌ Could not find {DB_FILE}")
        print("Please place the file in the same directory as this script or specify the path.")
        return
    
    print(f"📁 Found database: {db_path} ({db_path.stat().st_size / 1e6:.1f} MB)")
    
    # Create the dataset repo if it doesn't exist
    try:
        create_repo(REPO_ID, repo_type="dataset", exist_ok=True)
        print(f"✅ Dataset repo ready: https://huggingface.co/datasets/{REPO_ID}")
    except Exception as e:
        print(f"⚠️ Repo creation note: {e}")
    
    # Upload the file
    print("📤 Uploading database to Hugging Face Hub...")
    api.upload_file(
        path_or_fileobj=str(db_path),
        path_in_repo=DB_FILE,
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    
    print(f"✅ Upload complete!")
    print(f"🔗 Database URL: https://huggingface.co/datasets/{REPO_ID}/resolve/main/{DB_FILE}")

if __name__ == "__main__":
    main()
