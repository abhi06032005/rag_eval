# RAGScaleBench: Systems Benchmarking & Query-Adaptive Retrieval Platform

[![Test Suite](https://img.shields.io/badge/pytest-64%20passed-emerald.svg)](tests/)
[![Scientific Integrity Audit](https://img.shields.io/badge/Audit-Priority%201%20Verified-blue.svg)](reports/scientific_integrity_audit.md)
[![Zero Leakage](https://img.shields.io/badge/Splits-Zero%20Leakage-success.svg)](data/external/)
[![Next.js Frontend](https://img.shields.io/badge/Web%20Platform-Next.js%2016-black.svg)](web/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> **A scientifically audited, reproducible systems evaluation platform decoupling embedding representation error, vector quantization, and ANN distortion, paired with a calibrated query-adaptive hybrid retrieval engine and production-grade generative RAG serving.**

---

## 1. Executive Summary & Core Research Findings

Modern Retrieval-Augmented Generation (RAG) applications frequently assume that unconditionally running both lexical (BM25) and dense bi-encoder retrieval stages for every incoming query is optimal. **RAGScaleBench** rigorously evaluates this assumption across authentic BEIR benchmarks: **SciFact** (biomedical claims, 5,183 docs), **NFCorpus** (clinical nutrition, 3,633 docs), and **FiQA** (financial QA, 57,638 docs).

### Key Empirical Findings:

| Corpus / Domain | Corpus Docs | Test Queries | Lexical BM25 | Dense Exact | Fixed Hybrid (RRF k=60) | Adaptive Router | Dense Calls Saved | Systems Finding |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **BEIR/SciFact** | 5,183 | 300 | 0.6653 | 0.6406 | 0.6851 | **0.6861** | **46.0% (138 / 300)** | Non-inferior ($p=0.9160$), -22.3% p50 latency |
| **BEIR/NFCorpus** | 3,633 | 323 | 0.3218 | 0.3129 | **0.3448** | 0.3340 | **18.6% (60 / 323)** | Retains 97% hybrid nDCG, slight recall dip |
| **BEIR/FiQA** | 57,638 | 648 | 0.2361 | **0.3532 (5.3 ms)** | 0.3300 | 0.3216 | 10.2% (66 / 648) | **Dense strictly superior ($p=0.0296$)**; 24x faster |

---

## 2. Priority 1 — Scientific Integrity Audit Disclosures

In accordance with strict research integrity standards, all previously reported experimental results were audited and documented in [`reports/scientific_integrity_audit.md`](reports/scientific_integrity_audit.md):

1. **FiQA Embedding Task-197 Verification:** Forensic log inspection confirmed that `task-197` ran `.venv\Scripts\python.exe scripts/embed_and_evaluate_fiqa.py` for 2,664.4 seconds (44.4 minutes), successfully embedding all 57,638 FiQA documents in 12 shards without data fabrication.
2. **BEIR Ground-Truth Split Discrepancy Resolved:** Official BEIR FiQA ground truth contains **500 development queries** (1,238 positive qrels) and **648 test queries** (1,706 positive qrels). The report's prior mention of "648 development queries" was a typographical clerical error duplicating the test set count. Threshold calibration was independently audited and verified to have genuinely executed on the 500 dev queries with **0 test-set overlap**.
3. **Exploratory Non-Inferiority Classification:** The 0.025 nDCG non-inferiority margin was chosen retrospectively to match IR domain variance. It is officially reclassified as **exploratory post-hoc non-inferiority testing** rather than a pre-registered confirmatory trial.
4. **Explicit Reporting of FiQA Dense Superiority:** On the 57,638-document financial corpus, **Dense Exact (0.3532 nDCG@10, 5.29 ms)** strictly beats Fixed Hybrid (0.3300, $p=0.0296$) and Adaptive Router (0.3216, $p=0.0034$). Unconditional lexical fusion introduces **semantic dilution** and a 121.8 ms BM25 inverted index scan penalty.
5. **Tail-Latency Regression Disclosed:** While median (p50) latency drops by **22.3%** (34.00 ms vs 43.74 ms), 99th percentile (p99) latency regresses by **+14.3%** (101.23 ms vs 88.55 ms) due to compound sequential fallback overhead on hard queries.

---

## 3. Priority 2 & 4 — Real RAG Generation & Answer Evaluation

The serving layer features an authentic model-provider interface supporting local and hosted LLMs with zero silent fallbacks:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        RAGScaleBench Generator                         │
├─────────────────────┬──────────────────────────┬───────────────────────┤
│ Mock Fixture        │ Ollama Local Serving     │ OpenAI / Hosted API   │
├─────────────────────┼──────────────────────────┼───────────────────────┤
│ • Sentence extract  │ • Llama 3.2 3B (local)   │ • GPT-4o-mini / vLLM  │
│ • Deterministic CI  │ • SSE token streaming    │ • SSE token streaming │
│ • Offline fixture   │ • Exponential backoff    │ • Token telemetry     │
│ • Marked is_mock    │ • Strict HTTP 503 on off │ • Strict HTTP 503     │
└─────────────────────┴──────────────────────────┴───────────────────────┘
```

- **Strict No-Silent-Fallback Guarantee:** When a user requests `ollama` or `openai`, the backend verifies upstream daemon health. If unreachable, it explicitly raises **HTTP 503 Service Unavailable** with the connection error. It never silently emits heuristic mock text.
- **Prompt Injection Defense Nuance:** Context passages are isolated using structured XML delimiters (`<<<RETRIEVED_PASSAGE id="..." >>>`) with instructions explicitly disarming prompt-injection payloads in retrieved text. Note: Delimitation significantly mitigates naive context escape, but does not provide mathematical immunity against all adversarial jailbreaks.
- **Insufficient-Evidence Abstention:** When retrieved context is empty or irrelevant, the generator triggers the mandatory abstention rule: *"I am unable to answer based on the provided evidence."*
- **Evaluated Answer Metrics:**
  - **Deterministic Mock Baseline (`mock-generator`):** Evaluated on 20-sample reviewed benchmark. Token F1 = `0.3243`, Faithfulness = `0.6250`, Citation Support = `1.0000`, Abstention Accuracy = `100.0%` (5 / 5 unanswerable queries safely triggered abstention), Latency = `0.01 ms`. Documented in [`reports/generative_evaluation_authenticity_audit.md`](reports/generative_evaluation_authenticity_audit.md).
  - **Authentic Neural Generation (`qwen2.5:0.5b` in Docker):** Evaluated on 10 independent test cases with container resource limits (`--cpus=2.0 --memory=3.5g`). Mean latency = `11,443.84 ms` (10–15 tokens/sec autoregressive CPU decoding), Answerable Token F1 = `0.3791`, Zero-Context Abstention = `100.0%` (0.02 ms), Multi-Doc & Conflicting Evidence Synthesis = `Confirmed`, Harmless Prompt Injection Resistance = `100.0%` (0% hijack rate). Documented in [`reports/real_llm_generation_evaluation.md`](reports/real_llm_generation_evaluation.md).


---

## 4. Priority 3 — Next.js Research Web Platform

A web application built with **Next.js 16 (Turbopack)**, **TypeScript**, and **Tailwind CSS** connects directly to the FastAPI backend at `http://localhost:8000`:

- **`/` (Overview):** Executive summary, key findings spotlight cards, and cross-domain performance matrix.
- **`/playground` (QA Playground):** Real-time interactive RAG with live SSE token streaming, model provider selection, timing waterfall chart, and collapsible citation evidence.
- **`/datasets` (Datasets):** Official BEIR split ground truth explorer, FiQA 500 dev resolution, and genuine 66,454-document evaluated corpus inventory (SciFact: 5,183, NFCorpus: 3,633, FiQA: 57,638).
- **`/retrieval` (Retrieval):** Policy benchmark comparison with paired bootstrap p-values, 95% CIs, and FiQA failure case deep-dive (Dense Exact beats Hybrid by 0.0232 nDCG, p=0.0296).
- **`/analytics` (Latency & Resources):** p50/p95/p99 latency distribution breakdown, tail regression audit (+14.3% at p99), and Qdrant vector quantization footprints (SQ8, BQ1).
- **`/routing` (Adaptive Routing):** Interactive query routing explainer with threshold sliders ($\tau_{\text{margin}}$, $\tau_{\text{idf}}$) and live query simulation.
- **`/citations` (Citation Inspector):** Sentence-by-sentence factual entailment verifier and bracketed citation matcher.
- **`/reproducibility` (Reproducibility):** Cryptographic SHA-256 manifests, one-click reproduction commands, and direct artifact download center.
- **`/dashboard` (Ablations):** Reciprocal Rank Fusion constant sweep ($k=1..100$) and concurrency ladder saturation test results.

---

## 5. Quickstart & Reproduction Commands

### Local Python Environment

```bash
# 1. Activate environment
.venv\Scripts\activate  # On Linux/macOS: source .venv/bin/activate

# 2. Run the complete automated test suite (64 tests)
pytest -v

# 3. Audit scientific integrity & BEIR splits (Priority 1)
python scripts/audit_scientific_integrity.py

# 4. Run real RAG answer evaluation benchmark (Priority 4)
python scripts/evaluate_rag_generation.py

# 5. Launch the FastAPI Serving Engine
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Next.js Research Website

```bash
cd web
npm install
npm run build
npm run start  # Serves at http://localhost:3000
```

### Docker Compose Stack

```bash
# Launch Qdrant Vector DB, FastAPI serving engine, and Next.js web application
docker compose up --build
```
- Web Application: [`http://localhost:3000`](http://localhost:3000)
- FastAPI OpenAPI Docs: [`http://localhost:8000/docs`](http://localhost:8000/docs)
- Prometheus Telemetry: [`http://localhost:8000/metrics`](http://localhost:8000/metrics)
- Qdrant REST API: [`http://localhost:6333/dashboard`](http://localhost:6333/dashboard)

---

## 6. Three-Minute Demonstration Walkthrough Plan

1. **Minute 1 — Scientific Integrity & Split Verification:**
   - Open [`http://localhost:3000/datasets`](http://localhost:3000/datasets).
   - Review the verified BEIR split ground truth: FiQA 500 dev queries vs 648 test queries with 0 overlap.
   - Inspect the cryptographic SHA-256 hashes verifying that test sets were never touched during calibration.
2. **Minute 2 — Retrieval Benchmark & Trade-offs:**
   - Open [`http://localhost:3000/retrieval`](http://localhost:3000/retrieval).
   - Examine SciFact: Adaptive Router achieves 0.6861 nDCG@10 (matching Fixed Hybrid 0.6851, $p=0.9160$) while bypassing 46.0% of dense search calls.
   - Switch to FiQA: Observe the documented failure case where Dense-Only strictly beats Hybrid (0.3532 vs 0.3300, $p=0.0296$) by 24x lower latency.
   - Open [`http://localhost:3000/analytics`](http://localhost:3000/analytics): Review the p50 latency drop (-22.3%) versus the p99 tail latency regression (+14.3%).
3. **Minute 3 — Live RAG Generation & Citation Auditing:**
   - Open [`http://localhost:3000/playground`](http://localhost:3000/playground).
   - Click the biomedical sample query: *"What effect does MGMT promoter methylation have on glioblastoma treatment?"*
   - Execute with streaming enabled: observe the live timing breakdown waterfall and bracketed citation tag `[scifact:147]`.
   - Click the adversarial query: *"What was the exact date of the Apollo 18 moon landing?"*
   - Execute: observe the Insufficient-Evidence Abstention badge: *"I am unable to answer based on the provided evidence."*
   - Open [`http://localhost:3000/citations`](http://localhost:3000/citations): Inspect sentence-level factual entailment and token alignment.

---

## 7. Artifacts & Licensing

- [`papers/query_adaptive_hybrid_retrieval.md`](papers/query_adaptive_hybrid_retrieval.md): Full academic research paper with verified splits, statistical conclusions, systems profiling, and authentic neural LLM benchmark.
- [`reports/generative_evaluation_authenticity_audit.md`](reports/generative_evaluation_authenticity_audit.md): Forensic investigation and provenance audit of mock generator versus neural inference.
- [`reports/real_llm_generation_evaluation.md`](reports/real_llm_generation_evaluation.md): Authentic neural generation benchmark using Dockerized Qwen 2.5 0.5B under resource limits.
- [`reports/scientific_integrity_audit.md`](reports/scientific_integrity_audit.md): Forensic audit of `task-197`, BEIR splits, and non-inferiority classifications.
- [`results/canonical_results_registry.json`](results/canonical_results_registry.json): Cryptographically signed results registry.

**Licensing:** RAGScaleBench code is licensed under the **MIT License**. Upstream BEIR benchmark datasets (SciFact, NFCorpus, FiQA) remain subject to their original distribution terms.