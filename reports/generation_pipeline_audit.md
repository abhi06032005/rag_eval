# Forensic Audit of the RAG Generation Pipeline (`/ask`)
**RAGScaleBench — Phase 6 Research Deliverable (Task 6)**  
*Audit Date:* October 22, 2026  
*Status:* Completed Forensic Verification & Reality Disclosure  

---

## 1. Executive Summary & Core Disclosure
A rigorous empirical benchmark must clearly distinguish between **real neural language generation** and **offline heuristic test fixtures**. Prior reports listed citation precision and faithfulness figures without explicitly disclosing that the default pipeline operates on a deterministic template extractor rather than a real large language model (LLM).

### Mandated Disclosure:
1. **Responses originate from a Deterministic Heuristic Template:** By default (`GENERATOR_BACKEND=mock`), the `/ask` service uses [`MockGenerator`](file:///d:/ragscalebench/ragbench/generation/mock.py#L12-L17).
2. **Mechanism:** It splits the first retrieved passage by period (`.`), extracts the first non-empty sentence, and prefixes:
   `"Based on the provided evidence, {summary_sentence}. [{first_src}]"`
3. **No External LLM is Invoked:** Generating responses requires **0.0 GPU memory**, **0 LLM API calls**, and executes in **0.034 ms**.
4. **Exclusion from Production LLM Claims:** Generation experiments conducted with `MockGenerator` are strictly labeled as **offline smoke fixtures** and are explicitly excluded from production-generation quality claims.

---

## 2. Latency and Token Usage Separation

| Pipeline Stage | Fixed Hybrid Retrieval | LLM Generation (`MockGenerator`) | Production LLM (e.g., Llama-3-8B via vLLM) |
|---|---|---|---|
| **Latency p50** | ~34.0 ms | **0.024 ms** | ~400–1,200 ms (Inference Bound) |
| **Latency p95** | ~70.7 ms | **0.051 ms** | ~1,800–3,500 ms (TTFT + Decode) |
| **Token Usage (mean)** | N/A (Index Chunks) | **28.4 tokens** | 150–256 tokens |
| **Compute Bound** | CPU / RAM / IPC | **Pure CPU String Parsing** | GPU Tensor Core Memory Bandwidth |

In the default configuration, end-to-end latency in the `/ask` endpoint is **99.8% dominated by retrieval** (BM25 + vector search), with generation taking under 0.1 ms. When connected to a real LLM server (vLLM/Ollama), generation latency will dominate total response time by an order of magnitude.

---

## 3. Grounded Answer Quality & Claim-Level Entailment

We audited 20 reviewed scientific question-answering records using our claim decomposition and directional entailment engine ([`ragbench/evaluation/entailment.py`](file:///d:/ragscalebench/ragbench/evaluation/entailment.py)):

### 3.1 Citation Support vs. Factual Truth
- **Citation Precision:** 100.0%
- **Citation Recall:** 100.0%
- **Critical Finding:** A citation string pointing to a retrieved passage (e.g. `[doc:4387784]`) is **necessary but not sufficient** evidence of factual support. A model can cite an authentic retrieved paper while fabricating ungrounded claims.

### 3.2 Claim-Level Entailment Analysis
- **Total Atomic Claims Evaluated:** 20
- **Entailed Claims:** 20 (100.0%)
- **Unsupported Claims:** 0 (0.0%)
- Because `MockGenerator` extracts the verbatim leading sentence from the retrieved passage, entailed claim rates are high, but semantic nuance or contradictory evidence in subsequent sentences is missed.

### 3.3 Abstention Under Zero Context
When evaluated on queries with no retrieved context:
- `MockGenerator.generate(..., contexts=[])` returns:
  `"No context provided to answer: According to study 'CUA-Recovery: Benchmarking Error Recovery in Long-Horizon Computer-Use Agents', what finding or methodology is established?."`
- **Abstention Verification:** **PASSED**. The service safely abstains without fabricating pseudo-citations.
