"""
Controlled Concurrency & Load Testing Script for RAGScaleBench.
Evaluates vector retrieval latency percentiles and throughput under concurrency ladders.
Operates completely in-memory with deterministic embeddings, ensuring zero GPU/CPU overheating.
"""

import concurrent.futures
import json
import os
import time
import numpy as np
import pandas as pd

from ragbench.chunking import DocumentChunker
from ragbench.embeddings.sentence_transformers import OfflineHashedEmbedder
from ragbench.vectorstores.base import IndexConfig, SearchConfig, VectorRecord
from ragbench.vectorstores.qdrant import QdrantVectorStore


def run_controlled_load_test(
    corpus_parquet: str = "data/processed/documents.parquet",
    output_file: str = "results/load_test_results.json",
    queries_count: int = 100,
) -> dict:
    if not os.path.exists(corpus_parquet):
        raise FileNotFoundError(f"{corpus_parquet} not found")

    docs_df = pd.read_parquet(corpus_parquet)
    print(f"Loaded {len(docs_df)} documents for load testing.")

    # 1. Chunking
    chunker = DocumentChunker(chunk_size=256, overlap=32)
    all_chunks = []
    for _, doc in docs_df.iterrows():
        chunks = chunker.chunk_document(doc["doc_id"], doc["text"])
        all_chunks.extend(chunks)
    print(f"Generated {len(all_chunks)} chunks.")

    # 2. Embedding using deterministic fast embedder
    embedder = OfflineHashedEmbedder("mock-embedder", dimension=384)
    passage_vectors = embedder.embed_passages([c.text for c in all_chunks])

    # 3. Vector store setup
    vstore = QdrantVectorStore(in_memory=True)
    idx_cfg = IndexConfig(
        collection_name="load_test_collection",
        dimension=384,
        distance="cosine",
        m=16,
        ef_construct=128,
        quantization="none",
    )
    vstore.create_index(idx_cfg)

    records = [
        VectorRecord(id=c.chunk_id, vector=v, payload={"text": c.text, "doc_id": c.doc_id})
        for c, v in zip(all_chunks, passage_vectors)
    ]
    vstore.upsert(records)
    print("Vector index populated.")

    sample_queries = [
        "quantum computing and distributed systems architecture",
        "approximate nearest neighbor search in high dimensions",
        "regulatory compliance under federal statutory frameworks",
        "sentence respecting chunking and evidence span alignment",
        "vector quantisation trade-offs in low memory environments",
        "naval vessel engineering and propulsion systems",
        "information retrieval evaluation metrics precision and recall",
        "open source software licensing and copyleft obligations",
    ] * (queries_count // 8 + 1)
    sample_queries = sample_queries[:queries_count]
    query_vectors = [embedder.embed_query(q) for q in sample_queries]

    search_cfg = SearchConfig(k=10, hnsw_ef=64)

    # 4. Concurrency ladder sweep
    concurrency_ladder = [1, 2, 5, 10, 20]
    results = {}

    print("\nStarting Concurrency Load Sweep:")
    print(f"{'Concurrency':<12} | {'Queries':<8} | {'p50 Lat (ms)':<14} | {'p95 Lat (ms)':<14} | {'Throughput (RPS)':<16} | {'Errors':<8}")
    print("-" * 80)

    for conc in concurrency_ladder:
        latencies_ms = []
        errors = 0

        def worker(vec):
            t0 = time.perf_counter()
            try:
                hits = vstore.search(vec, search_cfg)
                dur = (time.perf_counter() - t0) * 1000.0
                return dur, True
            except Exception:
                dur = (time.perf_counter() - t0) * 1000.0
                return dur, False

        t_start_batch = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as executor:
            futures = [executor.submit(worker, v) for v in query_vectors]
            for f in concurrent.futures.as_completed(futures):
                dur, success = f.result()
                latencies_ms.append(dur)
                if not success:
                    errors += 1
        t_total_batch = time.perf_counter() - t_start_batch

        p50 = float(np.percentile(latencies_ms, 50))
        p95 = float(np.percentile(latencies_ms, 95))
        p99 = float(np.percentile(latencies_ms, 99))
        rps = len(query_vectors) / max(t_total_batch, 0.001)

        results[f"concurrency_{conc}"] = {
            "concurrency": conc,
            "queries": len(query_vectors),
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
            "throughput_rps": round(rps, 1),
            "errors": errors,
            "error_rate": round(errors / len(query_vectors), 4),
        }

        print(f"{conc:<12} | {len(query_vectors):<8} | {p50:<14.2f} | {p95:<14.2f} | {rps:<16.1f} | {errors:<8}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote load testing results to {output_file}")
    return results


if __name__ == "__main__":
    run_controlled_load_test()
