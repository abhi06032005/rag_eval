# RAGScaleBench — Three-Minute Demonstration Plan

This structured script guides an evaluator, reviewer, or stakeholder through an end-to-end, three-minute demonstration of RAGScaleBench's scientific rigor, systems telemetry, and real RAG product capabilities.

---

## Preparation (30 seconds before presentation)

1. Start the FastAPI backend:
   ```bash
   .venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
2. Start the Next.js web application (or run production build):
   ```bash
   cd web && npm run start
   ```
3. Open browser at [`http://localhost:3000`](http://localhost:3000). Ensure the top-right pill shows **"FastAPI :8000 Live"** in green.

---

## Minute 1: Scientific Rigor & BEIR Split Verification (0:00 – 1:00)

1. **Navigate to the Datasets Page (`/datasets`):**
   - **Show:** The BEIR Dataset Splits table.
   - **Explain:** *"Notice the ground truth numbers: SciFact has 809 train and 300 test queries. FiQA has 5,500 train, 500 dev, and 648 test queries."*
   - **Highlight Resolution:** *"Earlier drafts had a clerical typo repeating '648' for both dev and test. Our independent Priority 1 audit inspected execution logs and confirmed that threshold calibration ran strictly on the 500 dev queries, with zero query leakage into the test set."*
2. **Review Cryptographic Checksums:**
   - **Show:** The SHA-256 hashes verifying that test sets remained untouched and locked during all hyperparameter sweeps.

---

## Minute 2: Systems Retrieval Benchmark & Trade-offs (1:00 – 2:00)

1. **Navigate to the Retrieval Benchmark Page (`/retrieval`):**
   - **Show SciFact:** Point out that the Adaptive Router achieves **0.6861 nDCG@10**, matching Fixed Hybrid's 0.6851 ($p=0.9160$), while **eliminating 46.0% of dense bi-encoder embeddings and vector searches**.
   - **Explain Non-Inferiority Classification:** *"In accordance with scientific integrity, we do not claim equivalence. We reclassified our 0.025 nDCG non-inferiority margin as exploratory post-hoc analysis."*
2. **Show the Negative Finding (FiQA Failure Case):**
   - Click the **"FiQA-2018"** tab.
   - **Explain:** *"Unlike toy benchmarks that claim hybrid search is globally superior, RAGScaleBench exposes where hybrid fails. On FiQA's 57,638 financial documents, Dense-Only retrieval strictly beats Fixed Hybrid by +0.0232 nDCG ($p=0.0296$) and is 24x faster (5.3 ms vs 128 ms). Blending BM25 introduces severe semantic dilution."*
3. **Navigate to Latency & Resources (`/analytics`):**
   - **Show Tail Regression:** *"Notice our p50 median latency drops by 22.3%, but p99 tail latency regresses by +14.3% due to sequential fallback on difficult queries."*

---

## Minute 3: Generative RAG & Citation Auditing (2:00 – 3:00)

1. **Navigate to the QA Playground (`/playground`):**
   - **Execute Answerable Biomedical Query:** Click the first preloaded button (*"What effect does MGMT promoter methylation have on glioblastoma treatment?"*).
   - Click **"Execute RAG Pipeline"** with streaming enabled.
   - **Show:** Real-time token streaming, timing breakdown waterfall (Embedding -> Retrieval -> Context -> Generation), and the bracketed citation tag `[scifact:147]`.
   - **Expand Retrieved Evidence:** Show the exact retrieved paragraph with cosine similarity score.
2. **Execute Adversarial Abstention Query:**
   - Click the fourth preloaded button (*"What was the exact date and crew manifest of the Apollo 18 manned lunar landing mission?"*).
   - Click **"Execute RAG Pipeline"**.
   - **Show:** The **Abstained (Insufficient Evidence)** badge immediately triggers:
     > *"I am unable to answer based on the provided evidence."*
   - **Explain:** *"The system refuses to hallucinate facts when evidence is absent, achieving 100% abstention accuracy on our reviewed adversarial set."*
3. **Show No-Silent-Fallback Mode:**
   - Switch the provider tab to **"Local Ollama"** or **"OpenAI API"** (when offline).
   - Attempt execution: observe that the system returns an explicit, clearly labeled error rather than faking an answer with heuristic sentence extraction.

---

## Conclusion & Wrap-Up

Direct the audience to the **Reproducibility Page (`/reproducibility`)** for complete Docker commands, downloadable CSV/JSON manifests, and the full academic paper in `papers/query_adaptive_hybrid_retrieval.md`.
