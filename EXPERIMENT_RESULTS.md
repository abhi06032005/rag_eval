# RAGScaleBench Experimental Results & Pareto Analysis
**RAGScaleBench — Phase 4 Research Documentation**  
*Experiment Date:* October 2026  
*Hardware Environment:* AMD64 6-Core CPU, 16 GB RAM, Windows 11  

---

## 1. Official External Benchmark: BEIR SciFact End-to-End
*Dataset:* Official UKP Lab BEIR SciFact (5,183 authentic biomedical abstracts, 300 test claims, 339 human-annotated gold relevance pairs).  
*Embedding Model:* `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors, cached to disk).  
*Source Matrix:* [`results/scifact/results.csv`](file:///d:/ragscalebench/results/scifact/results.csv)  

| Configuration | Retrieval Backend | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | ANN Recall@10 | Mean Latency | Build Time |
|---|---|---|---|---|---|---|---|---|---|
| **BM25 (Lexical)** | Inverted Index (Pure Python) | 0.5150 | 0.7028 | 0.7605 | 0.6140 | 0.6432 | 1.0000 | 8.45 ms | 0.83 s |
| **Dense Exact (Brute Force)** | NumPy Cosine Matrix Product | 0.4823 | 0.7379 | 0.7833 | 0.6110 | 0.6451 | 1.0000 | 0.45 ms | 0.00 s |
| **Qdrant Standard HNSW** | Qdrant v1.19.2 ($m=16, ef=64$) | 0.4823 | 0.7379 | 0.7833 | 0.6110 | 0.6451 | **0.9970** | 15.07 ms | 4.62 s |
| **Qdrant Tuned HNSW** | Qdrant v1.19.2 ($m=32, ef=128$) | 0.4823 | 0.7379 | 0.7833 | 0.6110 | 0.6451 | **0.9990** | 15.15 ms | 5.36 s |
| **Qdrant Scalar Quant (INT8)**| Qdrant v1.19.2 (INT8 + Rescore) | 0.4790 | 0.7346 | 0.7800 | 0.6077 | 0.6417 | **0.9967** | 14.16 ms | 4.75 s |
| **Qdrant Binary Quant (1-bit)**| Qdrant v1.19.2 (1-bit + Rescore)| 0.4823 | 0.7379 | 0.7817 | 0.6105 | 0.6446 | **0.9803** | 13.40 ms | 4.93 s |
| **Hybrid (BM25 + Dense RRF)** | Reciprocal Rank Fusion ($k=60$) | **0.5298** | **0.7292** | **0.8004** | **0.6473** | **0.6763** | 1.0000 | 8.98 ms | 0.83 s |

### Key Findings (BEIR SciFact):
1. **Hybrid Retrieval Superiority:** Reciprocal Rank Fusion ($k=60$) combines the precise entity discrimination of BM25 with dense semantic generalization, achieving **Recall@10 of 0.8004** and **nDCG@10 of 0.6763** (+5.1% relative improvement over single-modality baselines).
2. **ANN Recall Separation:** By comparing against exact vector nearest neighbors, real approximation differences emerge: Standard HNSW achieves 0.9970, Tuned HNSW achieves 0.9990, Scalar Quant achieves 0.9967, and Binary Quant drops to 0.9803.

---

## 2. Real Qdrant Standalone Engine Audit
*Vector Engine:* Standalone native Rust binary Qdrant v1.19.2 (`bin/qdrant.exe`).  
*Collection Size:* 5,183 indexed points (384 dimensions).  
*Source Matrix:* [`results/real_qdrant/audit_metrics.json`](file:///d:/ragscalebench/results/real_qdrant/audit_metrics.json)  

| Collection Configuration | Status | Indexed Vectors | Segments | Actual Disk Footprint | Server RSS RAM |
|---|---|---|---|---|---|
| **Standard HNSW ($m=16, ef_c=100$)** | `green` | 5,183 | 2 | **138.95 MB** | **85.8 MB** |
| **Tuned HNSW ($m=32, ef_c=200$)** | `green` | 5,183 | 2 | **277.95 MB** | **96.3 MB** |
| **Scalar Quantization (INT8)** | `green` | 5,183 | 2 | **418.82 MB** | **108.9 MB** |
| **Binary Quantization (1-bit)** | `green` | 5,183 | 2 | **558.03 MB** | **121.4 MB** |

*Root Cause Confirmation:* In-memory embedded mode (`location=":memory:"`) executed brute-force matrix dot products, completely bypassing HNSW and quantization. In standalone server mode with `indexing_threshold=100`, real HNSW indexing and quantization rescoring were activated and verified via REST APIs.

---

## 3. Systems Concurrency & Saturation Curve
*Architecture:* Live containerized/standalone FastAPI gateway serving query requests.  
*Source Matrix:* [`results/load_tests/concurrency_sweep.json`](file:///d:/ragscalebench/results/load_tests/concurrency_sweep.json)  

| Concurrent Workers | Requests Completed | p50 Latency | p95 Latency | p99 Latency | Throughput (RPS) | Error Rate |
|---|---|---|---|---|---|---|
| **1 Worker** | 100 | 3.98 ms | 4.41 ms | 5.03 ms | **244.3 RPS** | 0.0% |
| **2 Workers** | 100 | 5.53 ms | 6.30 ms | 6.45 ms | **353.6 RPS** | 0.0% |
| **5 Workers (Peak Saturation)**| 100 | 11.49 ms | 13.75 ms | 14.61 ms | **421.3 RPS** | 0.0% |
| **10 Workers** | 100 | 21.93 ms | 31.51 ms | 36.57 ms | **415.9 RPS** | 0.0% |
| **20 Workers** | 100 | 36.56 ms | 63.37 ms | 70.96 ms | **401.3 RPS** | 0.0% |

*Key Systems Insight:* Physical saturation occurs at 5 concurrent client workers. Beyond 5 workers, queuing delay inflates p95 tail latency by 14.3x (from 4.41 ms to 63.37 ms).

---

## 4. Exploratory Synthetic Smoke-Test Matrix (Phase 2–3 Archive)
*Dataset:* 1,250 documents (500 arXiv, 500 Wikipedia, 250 GovInfo), 1,595 chunks.  
*Queries:* 50 synthetic test queries (exhibiting verbatim title leakage).  
*Mode:* Embedded In-Memory Vector Store (`LocalCollection`).  
*Classification:* Exploratory smoke tests only — NOT publication-grade evidence.  

| Configuration | Quantisation Mode | Recall@1 | Recall@10 | nDCG@10 | In-Memory p50 |
|---|---|---|---|---|---|
| **Baseline A (Exact Search)** | None (FP32) | 0.850 | 0.990 | 0.983 | 4.31 ms |
| **Baseline B (Default HNSW)** | None (FP32) | 0.850 | 0.990 | 0.983 | 4.92 ms |
| **Baseline C (Optimised HNSW)**| None (FP32) | 0.850 | 0.990 | 0.983 | 4.31 ms |
| **Baseline D (Scalar INT8)** | Scalar INT8 | 0.850 | 0.990 | 0.983 | 4.59 ms |
| **Baseline E (Binary 1-bit)** | Binary 1-bit | 0.850 | 0.990 | 0.983 | 4.24 ms |
