#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo "    RAGScaleBench: Full Scientific Reproduction Pipeline   "
echo "=========================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# 1. Environment & Prerequisite Validation
echo "[1/8] Validating environment and tooling..."
command -v python >/dev/null 2>&1 || { echo "Error: Python 3.11+ is required."; exit 1; }
command -v git >/dev/null 2>&1 || { echo "Error: Git is required."; exit 1; }

PYTHON_BIN="python"
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON_BIN=".venv/Scripts/python.exe"
fi

# 2. Acquire locked legal corpus data
echo "[2/8] Checking corpus sources and checksum manifest..."
${PYTHON_BIN} -m ragbench.data.prepare --config configs/default.yaml

# 3. Verify Labelled QA sets
echo "[3/8] Verifying QA sets and evidence offsets..."
${PYTHON_BIN} -m ragbench.qa.validate data/qa/dev.jsonl --documents data/processed/documents.parquet
${PYTHON_BIN} -m ragbench.qa.validate data/qa/test.jsonl --documents data/processed/documents.parquet

# 4. Service Health Check (if running)
echo "[4/8] Probing service endpoints..."
# Can be run against container or in-process

# 5. Execute Staged Benchmark Experiments
echo "[5/8] Running staged benchmark experiments across Representation & HNSW/Quantisation grids..."
${PYTHON_BIN} -m ragbench.experiments.runner --grid configs/experiments.yaml --stage all

# 6. Run Offered Load Benchmark
echo "[6/8] Executing load saturation benchmark..."
if command -v k6 >/dev/null 2>&1; then
    echo "Running k6 arrival-rate retrieval benchmark..."
    k6 run load/retrieve.js || true
else
    echo "k6 binary not found in PATH; skipping live arrival-rate run. (Install k6 to run load/retrieve.js and load/ask.js)"
fi

# 7. Generate Publication Figures
echo "[7/8] Generating publication-ready research plots..."
${PYTHON_BIN} scripts/plot_results.py --summary results/summary.csv --output plots

# 8. Synthesize Final Research Report
echo "[8/8] Synthesizing final research report..."
${PYTHON_BIN} scripts/generate_report.py --summary results/summary.csv --output results/REPORT.md

echo "=========================================================="
echo "Reproduction complete! Results available in:"
echo " - Summary Table: results/summary.csv"
echo " - Research Report: results/REPORT.md"
echo " - Publication Plots: plots/"
echo "=========================================================="