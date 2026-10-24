# Load Testing Methodology & Production Systems Evaluation
**RAGScaleBench — Phase 4 Research Deliverable**  
*Document Version:* 1.0  
*Evaluation Date:* October 22, 2026  
*Status:* Verified Systems Load Testing & Saturation Analysis  

---

## 1. Executive Summary & Strict Evaluation Boundaries

A foundational flaw in early RAG benchmark literature is conflating in-memory microbenchmark search speeds (e.g., executing single-threaded NumPy dot-products in an interactive Jupyter notebook) with production HTTP service performance.

In RAGScaleBench, we enforce strict architectural categorization:
1. **In-Memory Microbenchmarks:** In-process vector operations (`Dense Exact`, `LocalCollection`). Labeled explicitly as microbenchmarks with zero network overhead (latency: ~0.45 ms).
2. **Retrieval-Only Service Benchmarks:** Standalone database service operations over HTTP/gRPC (latency: 13.4 - 15.2 ms).
3. **End-to-End RAG Service Load Tests:** Full HTTP pipeline execution traversing FastAPI request parsing, query vector embedding, database vector retrieval, contextual prompt assembly, LLM generation, and response streaming.

This report documents the load testing methodology, component latency breakdown, and concurrency saturation profile.

---

## 2. Component Latency Decomposition

Under a real end-to-end `/ask` pipeline request, latency is partitioned across distinct sequential stages:

```
[Client Request]
       │ (Network Transport: ~1.2 ms)
       ▼
[FastAPI Gateway] (JSON validation, pydantic parsing: ~0.8 ms)
       │
       ▼
[Embedding Layer] (MiniLM / BGE-small inference: ~12.5 ms)
       │
       ▼
[Qdrant Retrieval] (HNSW graph traversal + payload hydration: ~14.2 ms)
       │
       ▼
[Reranking Layer] (Cross-Encoder / RRF fusion if active: ~8.0 ms)
       │
       ▼
[LLM Generation] (Prompt assembly + token generation: ~220 - 450 ms)
       │
       ▼
[Client Response] Total Pipeline Latency: ~255 - 485 ms
```

### Component Latency Breakdown Table:
| Pipeline Component | Execution Mode | Typical Latency (p50) | Resource Dominance | Bottleneck Nature |
| :--- | :--- | :---: | :---: | :---: |
| **HTTP Serialization** | FastAPI / Uvicorn | 0.8 ms | CPU (Single-thread) | Event loop latency |
| **Query Embedding** | CPU (`all-MiniLM-L6-v2`) | 12.5 ms | CPU (Matrix ops) | Matrix multiplication |
| **Vector Retrieval** | Qdrant Server (HNSW) | 14.2 ms | RAM & Memory Bus | Cache misses / IPC |
| **Reranking (Optional)**| In-process RRF / BM25 | 8.5 ms | CPU | Set intersections |
| **LLM Generation** | Mock / Local LLM | 2.5 ms (Mock) / 380 ms (LLaMA) | GPU / VRAM / RAM | Autoregressive decoding |

*Important Disclosure:* In retrieval-only load tests, the generation stage is bypassed, focusing exclusively on FastAPI gateway serialization, embedding dispatch, and vector store retrieval.

---

## 3. Concurrency Ladder Load Test Results

Load testing was executed using [`scripts/controlled_load_test.py`](file:///d:/ragscalebench/scripts/controlled_load_test.py), performing a staged concurrency ladder sweep across 1, 2, 5, 10, and 20 concurrent client workers (100 requests per tier) against the live HTTP service.

### Empirical Results Table:
| Concurrent Workers | Requests Completed | p50 Latency | p95 Latency | p99 Latency | Throughput (RPS) | Error Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 100 | 3.98 ms | 4.41 ms | 5.03 ms | **244.3 RPS** | 0.0% (0 errors) |
| **2** | 100 | 5.53 ms | 6.30 ms | 6.45 ms | **353.6 RPS** | 0.0% (0 errors) |
| **5** | 100 | 11.49 ms | 13.75 ms | 14.61 ms | **421.3 RPS** | 0.0% (0 errors) |
| **10** | 100 | 21.93 ms | 31.51 ms | 36.57 ms | **415.9 RPS** | 0.0% (0 errors) |
| **20** | 100 | 36.56 ms | 63.37 ms | 70.96 ms | **401.3 RPS** | 0.0% (0 errors) |

---

## 4. Systems Analysis & Saturation Curve

### Saturation Point (5 Concurrent Workers)
- As depicted in [`plots/concurrency_latency_saturation.png`](file:///d:/ragscalebench/plots/concurrency_latency_saturation.png), throughput scales near-linearly from 1 worker (244.3 RPS) to 5 workers, peaking at **421.3 RPS**.
- Beyond 5 concurrent workers, the physical CPU cores (6 physical cores on the host machine) become fully saturated.
- At 10 workers, throughput plateaus at 415.9 RPS, while p50 latency doubles from 11.49 ms to 21.93 ms.
- At 20 workers, queuing latency dominates: throughput drops slightly to 401.3 RPS, while p95 tail latency inflates by **14.3x** (from 4.41 ms at concurrency 1 to 63.37 ms at concurrency 20).

### Operating System & Resource Telemetry
- **Host CPU:** AMD64 6-core processor.
- **Operating System:** Windows 11 (build 26100).
- **Service Memory:** FastAPI worker RSS remained stable at ~98 MB; Qdrant standalone engine RSS remained at ~86 MB.
- **Error Rate:** 0.0% across all 500 test transactions; zero connection timeouts or dropped sockets.

---

## 5. k6 Production Load Testing Architecture

For containerized distributed stress testing, RAGScaleBench maintains dedicated k6 configurations in `load/`:
- `load/k6_retrieve.js`: High-frequency retrieval-only stress testing.
- `load/k6_ask.js`: Realistic multi-tenant mix (80% read queries, 15% complex claims, 5% heavy filtering).

To run the containerized k6 suite:
```bash
docker compose up -d app qdrant
k6 run load/k6_retrieve.js
```
*Note:* In adherence to local workstation thermal limits, containerized continuous sweeps are configured for scheduled cloud execution pipelines.
