# RAGScaleBench Benchmark Card
**RAGScaleBench — Phase 4 Research Documentation**  
*Document Version:* 2.0 (Post-Phase 4 Audit & External Validation)  

## 1. Benchmark Summary
- **Name:** RAGScaleBench
- **Version:** 0.2.0 (Phase 4 — Research-Grade Validation)
- **Purpose:** Systematic empirical evaluation of retrieval quality, vector quantisation degradation, indexing memory footprint, and query latency across approximate nearest neighbor (ANN) vector databases in production RAG serving architectures.
- **Context:** EDISS (European Master in Engineering of Data-Intensive Software Systems) Systems Research Portfolio.
- **Repository:** [`d:/ragscalebench`](file:///d:/ragscalebench)

---

## 2. Benchmark Tasks & Scope
RAGScaleBench isolates and measures the decoupled failure modes of retrieval-augmented generation pipelines:
1. **Embedding Model Representation Error:** Semantic gap between query representations and passage representations.
2. **Approximate Index Distortion:** Recall drop and rank permutation introduced by HNSW graph approximation, scalar quantisation (INT8), and binary quantisation (1-bit) relative to brute-force exact vector search.
3. **Lexical vs. Dense vs. Hybrid Orthogonality:** Information retrieval effectiveness of BM25 lexical matching versus dense semantic vectors and Reciprocal Rank Fusion ($k=60$).
4. **Authentic External Benchmark Validation:** Ground truth evaluation on the official **BEIR SciFact** benchmark (5,183 peer-reviewed biomedical abstracts, 300 atomic claim queries, 339 expert relevance judgments).
5. **Serving Performance & Saturation:** Query latency percentiles ($p50, p95, p99$), index build duration, throughput (RPS), queuing saturation under concurrency sweeps (1–20 workers), and physical on-disk storage sizes.

---

## 3. Evaluation Metrics

### A. Information Retrieval (IR) Quality
- **ANN Recall@10:** Exact nearest neighbor overlap between candidate HNSW retrieval and exact Euclidean/Cosine brute-force retrieval:
  $$\text{ANN-Recall@k} = \frac{|\text{Top-}k_{\text{approx}} \cap \text{Top-}k_{\text{exact}}|}{k}$$
- **Recall@k ($k \in \{1, 5, 10\}$):** Proportion of ground-truth evidence chunk IDs successfully retrieved within the top $k$ positions against official human labels.
- **Mean Reciprocal Rank (MRR):** Reciprocal rank of the first relevant evidence chunk.
- **nDCG@10:** Normalized Discounted Cumulative Gain accounting for rank position of gold evidence chunks.

### B. Generation & Faithfulness Quality
- **Exact Match (EM) & Token F1:** Normalized lexical overlap against ground truth answers.
- **Answer Correctness:** Semantic and factual correctness score against gold reference.
- **Context Precision & Recall:** Precision and recall of retrieved context containing necessary evidence spans.
- **Claim-Level Entailment Rate:** Percentage of generated claims verified as entailed by retrieved context passages.
- **Unsupported Claim Rate:** Rate of hallucinations / non-grounded claims.

### C. Systems & Infrastructure Metrics
- **Retrieval Latency ($p50, p95, p99$ in ms):** Measured strictly at the vector database search boundary (distinguishing in-memory microbenchmarks from HTTP socket requests).
- **Index Build Duration (seconds):** Total time to construct HNSW graph index and quantise vector representations.
- **Physical On-Disk Storage (MB):** Measured via recursive directory inspection on disk (including WAL logs, index pointers, and segment mappings).
- **Process Memory / RSS (MB):** Peak resident memory allocated for active vector storage and server runtimes.
- **Throughput (RPS):** Sustained query capacity under zero-error conditions across concurrency sweeps.

---

## 4. Hardware & Software Environment
- **CPU:** AMD64 6 physical cores
- **RAM:** 16 GB DDR4
- **Operating System:** Microsoft Windows 11
- **Python Environment:** CPython 3.11 with UV package manager
- **Vector Engine:** Qdrant v1.19.2 (Standalone Native Rust Server and Embedded Local Mode)
- **Embedder Models:**
  - `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, L2 normalized)
  - `BAAI/bge-small-en-v1.5` (384 dimensions, L2 normalized)
- **Generation Engine:** Deterministic Grounded Mock Generator (`mock-generator-v1`)

---

## 5. Methodological Taxonomy & Ethical Disclosures
- **Methodological Separation:**
  1. *Exploratory Smoke Tests:* Synthetic fixtures (50 questions with title-leakage) are strictly labeled as exploratory smoke tests.
  2. *Authentic External Benchmark:* BEIR SciFact (5,183 docs, 300 claims, 339 qrels) provides zero-leakage human-verified evaluation.
  3. *Qdrant Engine Audit:* Standalone Rust server v1.19.2 with verified HNSW index construction (`indexed_vectors_count: 5183`) and quantization rescoring.
  4. *HTTP Load Testing:* Real service concurrency ladder tests up to 20 workers.
- **Zero Fabrication Commitment:** No benchmark metrics, human verifications, or document counts are fabricated. All measurements derive from reproducible automated scripts.
