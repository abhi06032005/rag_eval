# RAGScaleBench — Complete Reproducibility Guide & Verification Recipe

This guide provides deterministic, step-by-step instructions to reproduce all empirical benchmarks, scientific integrity audits, generative evaluations, and web services in **RAGScaleBench**.

---

## 1. Environment Requirements

- **Python:** Version 3.11 or higher
- **Node.js:** Version 20 or higher (Node 24 tested) & `npm`
- **Operating System:** Windows 10/11, Ubuntu Linux 22.04+, or macOS
- **Hardware Minimums:** 8 GB RAM, 4 CPU cores, 5 GB free disk space
- **Hardware Recommended:** 16 GB RAM, 8 CPU cores

---

## 2. Installation & Quick Setup

```bash
# Clone the repository
git clone https://github.com/ragscalebench/ragscalebench.git
cd ragscalebench

# Create virtual environment and install dependencies
uv venv  # or python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

---

## 3. Priority 1 — Auditing Scientific Integrity & Splits

To independently verify the ground-truth BEIR splits, confirm task-197 execution, recalculate paired bootstrap confidence intervals, and audit zero train/dev/test query leakage:

```bash
python scripts/audit_scientific_integrity.py
```

Outputs:
- [`reports/scientific_integrity_audit.md`](reports/scientific_integrity_audit.md)
- [`results/scientific_integrity_audit.json`](results/scientific_integrity_audit.json)

---

## 4. Priority 4 — Evaluating Real RAG Generation & Abstention

To run the independent 20-sample reviewed evaluation benchmark measuring answer correctness, faithfulness, citation support, unsupported claims, and abstention accuracy:

```bash
python scripts/evaluate_rag_generation.py
```

Outputs:
- [`reports/rag_generation_evaluation.md`](reports/rag_generation_evaluation.md)
- [`results/qa/generation_evaluation_results.json`](results/qa/generation_evaluation_results.json)

---

## 5. Automated Test Suite (64 Tests)

To verify chunking, Qdrant vector storage, generator adapters, streaming endpoints, and zero silent mock fallbacks:

```bash
pytest -v
```

---

## 6. Launching Services & Research Website

### FastAPI Backend Service
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
- Swagger API Documentation: [`http://localhost:8000/docs`](http://localhost:8000/docs)
- Prometheus Telemetry: [`http://localhost:8000/metrics`](http://localhost:8000/metrics)

### Next.js Research Website
```bash
cd web
npm install
npm run build
npm run start
```
- Research Web Platform: [`http://localhost:3000`](http://localhost:3000)

---

## 7. Multi-Container Docker Deployment

To launch the full stack (Qdrant Rust Vector DB, FastAPI serving engine, and Next.js research website):

```bash
docker compose up -d --build
```
