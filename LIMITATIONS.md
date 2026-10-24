# Threats to Validity & Technical Limitations
**RAGScaleBench — Phase 4 Research Documentation**  
*Document Version:* 2.0 (Post-Phase 4 Audit & External Validation)  

This document explicitly identifies methodological limitations, hardware boundaries, and threats to internal and external validity in the **RAGScaleBench** project. In adherence to research ethics, all constraints are documented transparently.

---

## 1. Threats to Internal Validity

### 1.1 Synthetic vs. Authentic Evaluation Sets
- **Original Synthetic Test Partition (Phase 1–3):**
  - The 100 development questions and 50 locked test questions were produced via automated lexical span extraction.
  - *Identified Leakage:* 100% of the original synthetic test questions included the verbatim document title, allowing dense models to match surface title tokens rather than semantic evidence.
  - *Methodological Classification:* This partition is classified as an **exploratory smoke test set**, not publication-grade evidence.
- **Official BEIR SciFact Benchmark (Phase 4):**
  - To eliminate synthetic evaluation threats, Phase 4 implemented the official **BEIR SciFact** benchmark end-to-end (5,183 authentic biomedical abstracts, 300 atomic scientific claim queries, 339 human-annotated relevance judgments).
  - Queries are authentic claims (e.g., *"0-dimensional biomaterials lack inductive properties"*), eliminating title leakage completely.

### 1.2 In-Memory Search vs. Real Database Optimization Masking
- **The Local In-Memory Fallback Vector Store:**
  - In earlier iterations without a standalone server daemon, `QdrantVectorStore` defaulted to `QdrantClient(location=":memory:")`.
  - In local mode, Qdrant executes `qdrant_client.local.LocalCollection.search`, performing an **exhaustive brute-force NumPy matrix dot-product** over all vectors and ignoring `hnsw_ef`, `exact`, and `quantization_config`. This caused all 5 vector configurations to yield identical Recall@10 (0.990) and nDCG@10 (0.983).
- **Optimizer Threshold Masking:**
  - Qdrant defaults to `indexing_threshold = 20,000` vectors and `full_scan_threshold = 10,000` vectors. On collections under 10k items, Qdrant skips HNSW graph construction unless explicitly configured with lowered optimizer thresholds.
- **Resolution in Phase 4:**
  - Standalone Qdrant Rust engine v1.19.2 was deployed with `indexing_threshold = 100`.
  - HNSW index creation (`indexed_vectors_count: 5183`), INT8 scalar quantization, and 1-bit binary quantization were verified directly through Qdrant REST APIs.
  - ANN Recall@10 separation was verified: Standard HNSW (0.9970), Tuned HNSW (0.9990), Scalar Quant (0.9967), Binary Quant (0.9803).

---

## 2. Threats to External Validity & Scale Bounds

### 2.1 Vertical Scale & Thermal Envelopes
- **Verified Scale Envelope:**
  - Phase 4 verified scale: **5,183 authentic documents** (SciFact) and **1,250 multi-domain documents**.
  - All dense embeddings were cached to disk (`.npy`) using CPU thread restrictions (`threads=2`) to respect the workstation's thermal boundary constraints.
  - Full-scale execution at 25,000 and 75,000 documents requires distributed memory-mapped storage and batch pre-embedding on dedicated cloud instances (e.g. AWS c6i.2xlarge).

### 2.2 Storage Footprint: Theoretical Ratios vs. Physical Allocation
- **Theoretical Compression Fallacy:**
  - Theoretical binary quantization models assume a 32x reduction in vector memory ($384 \times 32\text{ bits} \to 384 \text{ bits}$).
  - *Physical Reality:* When `rescore=True` is enabled, Qdrant stores BOTH the 1-bit quantized vector index and the original 384-dimensional float32 vectors on disk for exact score recalculation. Additionally, Qdrant allocates 32 MB write-ahead log (WAL) segments per collection and HNSW graph pointer lists, meaning on-disk database size does not shrink proportionally to the in-RAM quantization ratio.

---

## 3. Systems Concurrency & Saturation Bounds

### 3.1 Core Contention under Load
- Load testing with `scripts/controlled_load_test.py` identified the physical saturation boundary on the host 6-core processor:
  - Throughput peaks at **5 concurrent client workers** (**421.3 RPS**).
  - Increasing concurrency from 1 to 20 workers increases p95 tail latency by **14.3x** (from 4.41 ms to 63.37 ms) due to request queuing and CPU context switching.
- *Strict Taxonomy Rule:* In-memory vector operations (0.45 ms) are classified as microbenchmarks and are never reported as production end-to-end RAG service performance.
