# Forensic Scientific Integrity Audit Report
**RAGScaleBench — Final Research Milestone**  
*Audit Date:* October 22, 2026  
*Status:* Verified, Audited, Scientific Discrepancies Formally Reconciled  

---

## 1. Executive Summary & Root-Cause Discrepancy Findings

This audit independently investigates and resolves scientific inconsistencies identified in previous milestone reports:

1. **FiQA Split Identifier Discrepancy (500 vs 648 Dev Queries):**
   - **Finding:** Official BEIR FiQA contains **500 development queries** (1,238 qrels) and **648 test queries** (1,706 qrels).
   - **Root Cause of Error:** Previous draft text erroneously repeated the number "648" for both development and test splits (`648 dev queries / 648 test queries`).
   - **Execution Verification:** Inspection of the actual execution code (`scripts/embed_and_evaluate_fiqa.py`) confirms that threshold calibration ran on `qrels/dev.tsv` (calibrating across all 500 dev queries), and evaluation ran on all 648 test queries. The discrepancy was a clerical reporting error in the markdown documentation, not an execution error.
   - **Train/Dev/Test Separation:** Pairwise set intersection confirms **0 query overlap** between train (5,500 queries), dev (500 queries), and test (648 queries) on FiQA, and 0 overlap on SciFact and NFCorpus.

2. **FiQA Task-197 Execution Authenticity:**
   - **Finding:** Task `task-197` executed `.venv\Scripts\python.exe scripts/embed_and_evaluate_fiqa.py`.
   - **Duration:** 2,664.4 seconds (44.4 minutes) under `torch.set_num_threads(3)`.
   - **Artifacts:** Verified on disk:
     - `results/fiqa/corpus_embeddings.npy` (88,532,096 bytes, shape 57,638 x 384, float32)
     - `results/fiqa/query_embeddings.npy` (995,456 bytes, shape 648 x 384, float32)
     - `results/fiqa/adaptive_routing_decisions.jsonl` (648 query routing records)
   - The reported FiQA results genuinely originated from this completed execution.

3. **Domain Degradation: Where Adaptive Retrieval is Worse than Dense-Only (FiQA):**
   - In financial opinion QA (FiQA), **Dense-Only retrieval is strictly superior to both Fixed Hybrid and Adaptive Retrieval in both quality and speed:**
     - Dense Exact: **nDCG@10 = 0.3532, Recall@10 = 0.4133, Mean Latency = 5.29 ms**
     - Fixed Hybrid: **nDCG@10 = 0.3300, Recall@10 = 0.4113, Mean Latency = 128.26 ms**
     - Adaptive Router: **nDCG@10 = 0.3216, Recall@10 = 0.4006, Mean Latency = 127.17 ms**
   - **Quality Degradation:** Adaptive retrieval is **statistically significantly worse than dense-only** ($\Delta = -0.0316$ nDCG, 95% CI $[-0.0537, -0.0099]$, $p = 0.0034$).
   - **Latency Inflation:** Adaptive retrieval is **24.0x slower than dense-only** (127.17 ms vs 5.29 ms) due to the invariant pure-Python inverted index probe over 57,638 documents.
   - **Scientific Explanation (Semantic Dilution):** In conversational finance, lexical keyword matching suffers severe vocabulary mismatch. Injecting noisy BM25 candidates into reciprocal rank fusion actively dilutes dense precision.

4. **Tail Latency Regression (p99) on SciFact:**
   - While the Adaptive Router improves median latency (p50: 34.00 ms vs 43.74 ms, -22.3%) and mean latency (34.49 ms vs 46.30 ms, -25.5%), **at the 99th percentile it regresses** (p99: **101.23 ms vs 88.55 ms, +14.3% slower**).
   - **Mechanism:** On the 1% tail of complex fallback queries, the router pays the full lexical search time + feature extraction + bi-encoder embedding + vector search + RRF fusion.

5. **Non-Inferiority Margin ($\delta = 0.025$) Reclassification:**
   - The $\delta = 0.025$ margin was introduced post-hoc in Phase 6 based on retrospective literature references (Voorhees 2000, Carterette 2007).
   - Because it was not prespecified in a registered protocol before observing Phase 5 test data, it is formally reclassified as **Exploratory Post-Hoc Non-Inferiority Analysis**.

---

## 2. Audited BEIR Dataset Split Manifest

| Dataset | Split | Queries with Gold Qrels | Qrel Pairs | Corpus Docs | Query Overlap with Other Splits | SHA-256 Checksum |
|---|---|---|---|---|---|---|
| **SciFact** | Train (Dev) | 809 | 919 | 5,183 | 0 queries overlap with Test | `a53f2114831916c0...` |
| **SciFact** | Test | 300 | 339 | 5,183 | 0 queries overlap with Train | `0864bb985e0ca236...` |
| **NFCorpus** | Train | 2,590 | 90,330 | 3,633 | 0 queries overlap with Dev/Test | `6336b80f9bffc4f0...` |
| **NFCorpus** | Dev | 324 | 12,148 | 3,633 | 0 queries overlap with Train/Test | `b1d38b5e8f78c4a5...` |
| **NFCorpus** | Test | 323 | 12,334 | 3,633 | 0 queries overlap with Train/Dev | `f8fba6ef3d4dd9c3...` |
| **FiQA** | Train | 5,500 | 14,166 | 57,638 | 0 queries overlap with Dev/Test | `e46a99529aa61b12...` |
| **FiQA** | Dev | **500** | 1,238 | 57,638 | 0 queries overlap with Train/Test | `03b27e547cd29dc7...` |
| **FiQA** | Test | **648** | 1,706 | 57,638 | 0 queries overlap with Train/Dev | `6adc2a640dcdd22b...` |

---

## 3. FiQA Cross-Domain Comparison: Dense Dominance & Failure Mode

| Configuration | Backend | Recall@10 | MRR | nDCG@10 | Latency (mean) | Latency (p50) | Dense Calls Saved |
|---|---|---|---|---|---|---|---|
| **BM25 (Okapi)** | Pure Python Inverted Index | 0.2843 | 0.2898 | 0.2247 | 121.80 ms | 121.57 ms | 100.0% (648/648) |
| **Dense Exact** | NumPy Cosine Matrix Dot Product | **0.4133** | **0.4382** | **0.3532** | **5.29 ms** | **5.04 ms** | 0.0% (0/648) |
| **Fixed Hybrid** | Reciprocal Rank Fusion ($k=60$) | 0.4113 | 0.4085 | 0.3300 | 128.26 ms | 126.94 ms | 0.0% (0/648) |
| **Adaptive Router** | Quality-Aware (tau_m=0.20, tau_idf=4.0) | 0.4006 | 0.4007 | 0.3216 | 127.17 ms | 125.43 ms | 10.2% (66/648) |

### Key Takeaways:
1. **Dense-Only Wins Decisively:** Dense retrieval is +0.0316 nDCG higher than Adaptive Router ($p=0.0034$) and 24x faster.
2. **Hybrid Inefficiency on Paraphrase Datasets:** Fixed Hybrid degrades dense quality by -0.0232 nDCG ($p=0.0296$) because noisy BM25 top hits displace true dense hits under RRF.
3. **Adaptive Router Safety:** The router safely backs off to dense/hybrid on 89.8% of queries, saving 10.2% of calls on exact queries. However, because it incurs the 121 ms lexical search on every query, adaptive routing is economically unviable on FiQA when dense matrix multiplication on CPU takes only 5.29 ms.

---

## 4. Latency Distribution & Tail Regression (SciFact)

| Metric | Fixed Hybrid | Adaptive Router | Delta | Operational Implication |
|---|---|---|---|---|
| **p50 Latency** | 43.74 ms | **34.00 ms** | **-9.74 ms (-22.3%)** | Marked median serving latency reduction |
| **Mean Latency**| 46.30 ms | **34.49 ms** | **-11.81 ms (-25.5%)** | Net overall throughput improvement (+34.2%) |
| **p95 Latency** | 72.31 ms | **70.69 ms** | **-1.62 ms (-2.2%)** | Modest 95th percentile latency reduction |
| **p99 Latency** | **88.55 ms** | **101.23 ms** | **+12.68 ms (+14.3%)** | **Tail latency regression on complex fallback queries** |

---

## 5. Statistical Integrity Summary & Reclassification

1. **Equivalence Claim:** Replaced with "no statistically significant difference observed" ($\Delta = -0.0062, p=0.2214$).
2. **Non-Inferiority Testing:** Formally labeled **Exploratory Post-Hoc Analysis**. Rejection of $H_0$ ($p < 0.0001$) holds mathematically for $\delta=0.025$, but cannot be claimed as a prespecified confirmatory endpoint.
3. **Multiple Testing Adjustments:** Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR) adjustments confirm that BM25 vs Hybrid ($p=0.0096$ Holm) and BM25 vs Adaptive ($p=0.0006$ Holm) remain statistically significant, while Hybrid vs Adaptive ($p=0.4428$ Holm) is not statistically significant.
