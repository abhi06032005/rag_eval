# Research Methodology & Experimental Design
**RAGScaleBench — Phase 4 Research Documentation**  
*Document Version:* 2.0 (Post-Phase 4 Audit & External Validation)  

---

## 1. Research Questions (RQs)

The central objective of **RAGScaleBench** is to quantify the trade-offs between retrieval accuracy, graph indexing latency, memory consumption, and downstream generation faithfulness in production RAG systems.

We formulate five core systems-research questions:

- **RQ1 (Quantisation vs. Quality):** How severely does vector quantisation (Scalar INT8 and Binary 1-bit with rescoring) degrade ANN neighbour recall and human-judged gold evidence retrieval compared to unquantised 32-bit floating point vectors?
- **RQ2 (Graph Structure vs. Latency):** What is the empirical latency–recall curve when varying HNSW construction parameters ($M \in \{16, 32\}$, $ef_{\text{construct}} \in \{100, 200\}$) and query-time candidate list depth ($ef \in \{64, 128\}$)?
- **RQ3 (Lexical vs. Dense vs. Hybrid Fusion):** To what extent does Reciprocal Rank Fusion (RRF) combining lexical BM25 and dense neural representations outperform single-modality retrievers on authentic peer-reviewed claims?
- **RQ4 (Theoretical vs. Physical Vector Footprint):** How does real on-disk storage (incorporating WAL logs, segment fragmentation, and HNSW graph links) diverge from theoretical vector compression formulas?
- **RQ5 (Concurrency Saturation & Latency Tail Inflation):** Where is the physical saturation knee of the serving architecture, and how severely does tail latency (p95/p99) inflate under concurrent load?

---

## 2. Experimental Taxonomy & Methodological Tiers

In strict compliance with empirical systems research standards, RAGScaleBench categorizes experimental evaluations into four mutually exclusive tiers to avoid conflating different methodologies:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       RAGScaleBench Experimental Tiers                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier A: Exploratory Synthetic Smoke Tests (1,250 docs, 50 queries)          │
│   - Rapid CI pipeline validation & regression testing.                      │
│   - Characterized by title-leakage and embedded in-memory execution.        │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier B: Authentic External Benchmark (BEIR SciFact, 5,183 docs, 300 claims) │
│   - Zero title-leakage; authentic human-annotated scientific ground truth.  │
│   - Evaluates BM25, Dense Exact, Standalone Qdrant, and Hybrid RRF.         │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier C: Vector Database Engine Audit (Standalone Qdrant v1.19.2)            │
│   - Direct REST API verification of indexed_vectors_count & collection info.│
│   - Measures ANN Recall@10 vs. Exact Nearest Neighbors.                     │
│   - Records real on-disk directory sizes and process RSS memory.            │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier D: Systems HTTP Load Testing & Saturation Sweep                        │
│   - Live FastAPI + Qdrant service testing via concurrency ladder sweeps.    │
│   - Identifies saturation knee (5 workers, 421.3 RPS) and tail inflation.   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Retrieval Baseline Architectures

To establish fair, reproducible comparisons, all models share matched embeddings, document collections, query partitions, and hardware constraints:

1. **BM25 Okapi (Lexical Baseline):**
   - Pure-Python inverted index with Robertson-Spärck Jones IDF smoothing ($k_1=1.5, b=0.75$).
2. **Dense Exact (Brute Force Ground Truth):**
   - Exhaustive float32 matrix cosine dot-product ($Q \cdot C^T$).
   - Serves as the ground truth for ANN Recall calculation ($\text{ANN Recall} \equiv 1.0000$).
3. **Qdrant Standard HNSW:**
   - Standalone Rust Qdrant server v1.19.2: $M=16, ef_{\text{construct}}=100, ef_{\text{search}}=64$.
4. **Qdrant Tuned HNSW:**
   - High-connectivity graph: $M=32, ef_{\text{construct}}=200, ef_{\text{search}}=128$.
5. **Qdrant Scalar Quantization (INT8 + Rescore):**
   - 8-bit integer quantization with quantile calibration (0.99) and exact float32 rescoring.
6. **Qdrant Binary Quantization (1-bit + Rescore):**
   - 1-bit binary projection with $2\times$ candidate oversampling and full-vector rescoring.
7. **Hybrid RRF (Lexical + Dense):**
   - Reciprocal Rank Fusion ($k_{rrf}=60$) combining BM25 and Dense Exact/HNSW ranks without test-set tuning.

---

## 4. Systems Metrics Protocol & Boundary Enforcement

- **Microbenchmarks vs. Service Benchmarks:**
  - Microbenchmarks (in-process matrix multiplications taking 0.45 ms) are strictly labeled as microbenchmarks.
  - Network service benchmarks (HTTP/gRPC traversals taking ~14 ms) reflect real IPC, serialization, and database server execution.
- **Thermal & Resource Safety Boundaries:**
  - Neural dense embeddings are computed in batches with restricted CPU threads (`torch.set_num_threads(2)`) and cached to disk as `.npy` arrays, eliminating repeated thermal loads.
- **Process Lifecycle Cleanliness:**
  - Standalone database servers (e.g., `bin/qdrant.exe`) are managed via strict `try...finally` blocks, guaranteeing termination upon experiment completion (zero orphaned daemons).
- **Physical Memory Accounting:**
  - Process Resident Set Size (RSS) is queried via `psutil.Process().memory_info().rss`.
  - Disk storage is computed via recursive filesystem directory inspection (`get_dir_size_bytes()`), never through theoretical multiplication formulas.
