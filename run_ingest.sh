#!/usr/bin/env bash
# run_ingest.sh — convenience wrapper that sets PYTHONPATH correctly.
# Usage:
#   ./run_ingest.sh                 # full index (PDFs + job listings)
#   ./run_ingest.sh --skip-jobs     # PDFs only
#   ./run_ingest.sh --skip-pdfs     # job listings only
#   ./run_ingest.sh --job-limit 500 # smoke-test: first 500 jobs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/../.venv/bin/python"

# Fall back to system python if .venv doesn't exist
if [[ ! -f "$PYTHON" ]]; then
    PYTHON="$(which python3 || which python)"
fi

PYTHONPATH="$SCRIPT_DIR" "$PYTHON" "$SCRIPT_DIR/ingest/build_index.py" "$@"
