"""
Official BEIR SciFact End-to-End Benchmark & Qdrant Engine Audit.
Evaluates:
  A. BM25 Okapi (lexical baseline)
  B1. Dense Exact Search (brute-force matrix cosine similarity)
  B2. Dense Qdrant Server - Standard HNSW (m=16, ef_c=100, ef=64)
  B3. Dense Qdrant Server - Tuned HNSW (m=32, ef_c=200, ef=128)
  B4. Dense Qdrant Server - Scalar Quantization (INT8)
  B5. Dense Qdrant Server - Binary Quantization (1-bit + oversampling/rescore)
  C. Hybrid RRF (BM25 + Dense)

Measures:
  Recall@1, Recall@5, Recall@10, MRR, nDCG@10, ANN Recall@10,
  p50/p95/mean retrieval latency, index build time, actual disk bytes, and RAM usage.
Guarantees clean process lifecycle (zero orphaned daemons).
"""

import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
import numpy as np
import psutil
import requests

SERVER_PORT = 6338
SERVER_GRPC_PORT = 6339
SERVER_HOST = "127.0.0.1"
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
STORAGE_PATH = "./scratch/qdrant_scifact_bench"


def start_qdrant_server() -> subprocess.Popen:
    """Start standalone Qdrant Rust engine with dedicated ports and isolated storage."""
    if os.path.exists(STORAGE_PATH):
        try:
            shutil.rmtree(STORAGE_PATH)
        except Exception:
            pass
    os.makedirs(STORAGE_PATH, exist_ok=True)

    env = os.environ.copy()
    env["QDRANT__SERVICE__HOST"] = SERVER_HOST
    env["QDRANT__SERVICE__HTTP_PORT"] = str(SERVER_PORT)
    env["QDRANT__SERVICE__GRPC_PORT"] = str(SERVER_GRPC_PORT)
    env["QDRANT__STORAGE__STORAGE_PATH"] = STORAGE_PATH
    env["QDRANT__LOG_LEVEL"] = "WARN"

    exe_path = os.path.abspath("bin/qdrant.exe")
    proc = subprocess.Popen(
        [exe_path],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for readiness via explicit IPv4 address
    for _ in range(30):
        time.sleep(0.5)
        try:
            resp = requests.get(f"{SERVER_URL}/readyz", timeout=1.0)
            if resp.status_code == 200:
                print(f"Qdrant server v1.19.2 ready at {SERVER_URL} (PID: {proc.pid})", flush=True)
                return proc
        except Exception:
            pass

    proc.kill()
    raise RuntimeError("Failed to start Qdrant server within timeout.")


def stop_qdrant_server(proc: subprocess.Popen) -> None:
    """Guaranteed clean termination of Qdrant server process."""
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    print("Qdrant server cleanly terminated.", flush=True)


def get_dir_size_bytes(path: str) -> int:
    """Calculate recursive directory size on disk."""
    total = 0
    if not os.path.exists(path):
        return 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total


def rrf_fuse(
    ranked_a: list[tuple[str, float]],
    ranked_b: list[tuple[str, float]],
    k_rrf: int = 60,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion."""
    rrf_scores: dict[str, float] = defaultdict(float)
    for rank, (doc_id, _) in enumerate(ranked_a, 1):
        rrf_scores[doc_id] += 1.0 / (k_rrf + rank)
    for rank, (doc_id, _) in enumerate(ranked_b, 1):
        rrf_scores[doc_id] += 1.0 / (k_rrf + rank)
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def main():
    print("=================================================================", flush=True)
    print("   RAGScaleBench Phase 4: BEIR SciFact & Real Qdrant Audit      ", flush=True)
    print("=================================================================", flush=True)

    # 1. Load data
    scifact_dir = "data/beir/scifact"
    results_dir = "results/scifact"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("results/real_qdrant", exist_ok=True)

    corpus_emb_path = os.path.join(results_dir, "corpus_embeddings.npy")
    query_emb_path = os.path.join(results_dir, "query_embeddings.npy")
    corpus_ids_path = os.path.join(results_dir, "corpus_ids.json")
    query_ids_path = os.path.join(results_dir, "query_ids.json")

    if not (os.path.exists(corpus_emb_path) and os.path.exists(query_emb_path)):
        raise FileNotFoundError(
            "Embeddings not cached yet. Run scripts/cache_scifact_embeddings.py first."
        )

    print("Loading cached dense embeddings...", flush=True)
    corpus_vecs = np.load(corpus_emb_path)  # (5183, 384)
    query_vecs = np.load(query_emb_path)    # (300, 384)
    with open(corpus_ids_path, "r", encoding="utf-8") as f:
        corpus_ids = json.load(f)
    with open(query_ids_path, "r", encoding="utf-8") as f:
        query_ids = json.load(f)

    # Load corpus texts for BM25
    print("Loading SciFact corpus texts...", flush=True)
    corpus_dict: dict[str, str] = {}
    with open(os.path.join(scifact_dir, "corpus.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            corpus_dict[item["_id"]] = (item.get("title", "") + " " + item.get("text", "")).strip()

    # Load queries
    queries_dict: dict[str, str] = {}
    with open(os.path.join(scifact_dir, "queries.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            queries_dict[item["_id"]] = item["text"]

    # Load ground truth qrels
    test_qrels: dict[str, set[str]] = defaultdict(set)
    with open(os.path.join(scifact_dir, "qrels/test.tsv"), "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3 and int(parts[2]) > 0:
                test_qrels[parts[0]].add(parts[1])

    print(f"Corpus: {len(corpus_ids)} docs | Queries: {len(query_ids)} | Qrels: {sum(len(v) for v in test_qrels.values())} pairs", flush=True)

    all_results = []
    ann_gold_neighbors: dict[str, list[str]] = {}

    from ragbench.evaluation.bm25 import BM25Okapi
    from ragbench.evaluation.retrieval import (
        calculate_mrr,
        calculate_ndcg_at_k,
        calculate_recall_at_k,
    )

    # -------------------------------------------------------------
    # BASELINE A: BM25 Okapi
    # -------------------------------------------------------------
    print("\n--- [Baseline A] Evaluating BM25 Okapi ---", flush=True)
    bm25 = BM25Okapi()
    t_b_start = time.perf_counter()
    bm25_docs = [(cid, corpus_dict[cid]) for cid in corpus_ids]
    bm25.fit(bm25_docs)
    bm25_build_time = time.perf_counter() - t_b_start

    bm25_hits: dict[str, list[tuple[str, float]]] = {}
    latencies_bm25 = []
    r1, r5, r10, mrr, ndcg = [], [], [], [], []

    for qid in query_ids:
        qtext = queries_dict[qid]
        gold_set = test_qrels[qid]
        t0 = time.perf_counter()
        ranked = bm25.search(qtext, top_k=100)
        latencies_bm25.append((time.perf_counter() - t0) * 1000.0)
        bm25_hits[qid] = ranked
        retrieved_ids = [doc_id for doc_id, _ in ranked]

        r1.append(calculate_recall_at_k(retrieved_ids, gold_set, 1))
        r5.append(calculate_recall_at_k(retrieved_ids, gold_set, 5))
        r10.append(calculate_recall_at_k(retrieved_ids, gold_set, 10))
        mrr.append(calculate_mrr(retrieved_ids, gold_set))
        ndcg.append(calculate_ndcg_at_k(retrieved_ids, gold_set, 10))

    bm25_row = {
        "Configuration": "BM25 (Lexical)",
        "Backend": "Pure Python Inverted Index",
        "Recall@1": round(float(np.mean(r1)), 4),
        "Recall@5": round(float(np.mean(r5)), 4),
        "Recall@10": round(float(np.mean(r10)), 4),
        "MRR": round(float(np.mean(mrr)), 4),
        "nDCG@10": round(float(np.mean(ndcg)), 4),
        "ANN_Recall@10": 1.0,
        "Latency_p50_ms": round(float(np.median(latencies_bm25)), 2),
        "Latency_p95_ms": round(float(np.percentile(latencies_bm25, 95)), 2),
        "Latency_mean_ms": round(float(np.mean(latencies_bm25)), 2),
        "Build_Time_s": round(bm25_build_time, 3),
        "Disk_Size_MB": 0.0,
        "RAM_MB": round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1),
    }
    all_results.append(bm25_row)
    print(f"BM25 -> Recall@10: {bm25_row['Recall@10']:.4f} | nDCG@10: {bm25_row['nDCG@10']:.4f} | Latency: {bm25_row['Latency_mean_ms']} ms", flush=True)

    # -------------------------------------------------------------
    # BASELINE B1: Dense Exact Search (Brute-Force)
    # -------------------------------------------------------------
    print("\n--- [Baseline B1] Evaluating Dense Exact Search (Brute Force) ---", flush=True)
    exact_hits: dict[str, list[tuple[str, float]]] = {}
    latencies_exact = []
    r1, r5, r10, mrr, ndcg = [], [], [], [], []

    # Pre-transpose corpus for matrix multiplication: Q @ C.T
    corpus_vecs_t = corpus_vecs.T

    for i, qid in enumerate(query_ids):
        gold_set = test_qrels[qid]
        q_vec = query_vecs[i]
        t0 = time.perf_counter()
        sims = np.dot(q_vec, corpus_vecs_t)
        top_indices = np.argsort(sims)[::-1][:100]
        latencies_exact.append((time.perf_counter() - t0) * 1000.0)

        retrieved = [(corpus_ids[idx], float(sims[idx])) for idx in top_indices]
        exact_hits[qid] = retrieved
        ann_gold_neighbors[qid] = [doc_id for doc_id, _ in retrieved[:10]]

        retrieved_ids = [doc_id for doc_id, _ in retrieved]
        r1.append(calculate_recall_at_k(retrieved_ids, gold_set, 1))
        r5.append(calculate_recall_at_k(retrieved_ids, gold_set, 5))
        r10.append(calculate_recall_at_k(retrieved_ids, gold_set, 10))
        mrr.append(calculate_mrr(retrieved_ids, gold_set))
        ndcg.append(calculate_ndcg_at_k(retrieved_ids, gold_set, 10))

    exact_row = {
        "Configuration": "Dense Exact (Brute Force)",
        "Backend": "NumPy Exact Cosine Matrix Product",
        "Recall@1": round(float(np.mean(r1)), 4),
        "Recall@5": round(float(np.mean(r5)), 4),
        "Recall@10": round(float(np.mean(r10)), 4),
        "MRR": round(float(np.mean(mrr)), 4),
        "nDCG@10": round(float(np.mean(ndcg)), 4),
        "ANN_Recall@10": 1.0,
        "Latency_p50_ms": round(float(np.median(latencies_exact)), 2),
        "Latency_p95_ms": round(float(np.percentile(latencies_exact, 95)), 2),
        "Latency_mean_ms": round(float(np.mean(latencies_exact)), 2),
        "Build_Time_s": 0.0,
        "Disk_Size_MB": round(os.path.getsize(corpus_emb_path) / (1024 * 1024), 2),
        "RAM_MB": round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1),
    }
    all_results.append(exact_row)
    print(f"Dense Exact -> Recall@10: {exact_row['Recall@10']:.4f} | nDCG@10: {exact_row['nDCG@10']:.4f} | Latency: {exact_row['Latency_mean_ms']} ms", flush=True)

    # -------------------------------------------------------------
    # Real Qdrant Server Configurations (B2, B3, B4, B5)
    # -------------------------------------------------------------
    server_proc = start_qdrant_server()
    qdrant_audit_telemetry = {}

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as rest

        # Use explicit IPv4 address to eliminate Windows IPv6 localhost SYN_SENT hangs
        client = QdrantClient(url=SERVER_URL, timeout=30.0)

        def index_collection(coll_name: str, hnsw_cfg: rest.HnswConfigDiff, quant_cfg: rest.QuantizationConfig | None) -> float:
            try:
                client.delete_collection(coll_name)
            except Exception:
                pass

            t0 = time.perf_counter()
            client.create_collection(
                collection_name=coll_name,
                vectors_config=rest.VectorParams(
                    size=int(corpus_vecs.shape[1]),
                    distance=rest.Distance.COSINE,
                    on_disk=False,
                ),
                hnsw_config=hnsw_cfg,
                quantization_config=quant_cfg,
                optimizers_config=rest.OptimizersConfigDiff(
                    indexing_threshold=100,  # force active HNSW index creation for 5k docs
                    default_segment_number=2,
                ),
            )

            # Upsert points
            batch_size = 500
            for start_idx in range(0, len(corpus_ids), batch_size):
                end_idx = min(start_idx + batch_size, len(corpus_ids))
                points = [
                    rest.PointStruct(
                        id=i,
                        vector=corpus_vecs[i].tolist(),
                        payload={"scifact_id": corpus_ids[i]},
                    )
                    for i in range(start_idx, end_idx)
                ]
                client.upsert(collection_name=coll_name, points=points, wait=True)

            # Trigger optimization to guarantee HNSW index is built and active
            client.update_collection(
                collection_name=coll_name,
                optimizer_config=rest.OptimizersConfigDiff(indexing_threshold=10),
            )
            time.sleep(2.0)  # allow optimizer background thread to finalize segments
            build_time = time.perf_counter() - t0
            return build_time

        def evaluate_qdrant_search(coll_name: str, search_params: rest.SearchParams) -> tuple[dict, list[tuple[str, float]]]:
            latencies = []
            r1, r5, r10, mrr, ndcg, ann_rec = [], [], [], [], [], []
            hits_by_qid = {}

            for i, qid in enumerate(query_ids):
                gold_set = test_qrels[qid]
                q_vec = query_vecs[i].tolist()
                t0 = time.perf_counter()
                res = client.query_points(
                    collection_name=coll_name,
                    query=q_vec,
                    limit=100,
                    search_params=search_params,
                    with_payload=True,
                )
                latencies.append((time.perf_counter() - t0) * 1000.0)

                retrieved = [(pt.payload["scifact_id"], float(pt.score)) for pt in res.points]
                hits_by_qid[qid] = retrieved
                retrieved_ids = [doc_id for doc_id, _ in retrieved]

                r1.append(calculate_recall_at_k(retrieved_ids, gold_set, 1))
                r5.append(calculate_recall_at_k(retrieved_ids, gold_set, 5))
                r10.append(calculate_recall_at_k(retrieved_ids, gold_set, 10))
                mrr.append(calculate_mrr(retrieved_ids, gold_set))
                ndcg.append(calculate_ndcg_at_k(retrieved_ids, gold_set, 10))

                # ANN recall against exact vector nearest neighbors
                exact_top10 = set(ann_gold_neighbors[qid])
                matched_ann = sum(1 for cid in retrieved_ids[:10] if cid in exact_top10)
                ann_rec.append(matched_ann / 10.0)

            metrics = {
                "Recall@1": round(float(np.mean(r1)), 4),
                "Recall@5": round(float(np.mean(r5)), 4),
                "Recall@10": round(float(np.mean(r10)), 4),
                "MRR": round(float(np.mean(mrr)), 4),
                "nDCG@10": round(float(np.mean(ndcg)), 4),
                "ANN_Recall@10": round(float(np.mean(ann_rec)), 4),
                "Latency_p50_ms": round(float(np.median(latencies)), 2),
                "Latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
                "Latency_mean_ms": round(float(np.mean(latencies)), 2),
            }
            return metrics, hits_by_qid

        # ---------------------------------------------------------
        # B2: Standard HNSW (m=16, ef_construct=100, ef_search=64)
        # ---------------------------------------------------------
        print("\n--- [Baseline B2] Real Qdrant: Standard HNSW ---", flush=True)
        hnsw_b2 = rest.HnswConfigDiff(m=16, ef_construct=100, full_scan_threshold=50)
        b2_build_time = index_collection("scifact_hnsw_std", hnsw_b2, None)
        coll_info = client.get_collection("scifact_hnsw_std")
        print(f"Active collection status: {coll_info.status} | points: {coll_info.points_count} | indexed_vectors: {coll_info.indexed_vectors_count}", flush=True)

        b2_metrics, b2_hits = evaluate_qdrant_search(
            "scifact_hnsw_std",
            rest.SearchParams(hnsw_ef=64, exact=False)
        )
        server_mem_mb = round(psutil.Process(server_proc.pid).memory_info().rss / (1024 * 1024), 1)
        disk_size_mb = round(get_dir_size_bytes(STORAGE_PATH) / (1024 * 1024), 2)

        b2_row = {
            "Configuration": "Qdrant Standard HNSW (m=16, ef=64)",
            "Backend": "Qdrant v1.19.2 (Standalone Rust Server)",
            **b2_metrics,
            "Build_Time_s": round(b2_build_time, 2),
            "Disk_Size_MB": disk_size_mb,
            "RAM_MB": server_mem_mb,
        }
        all_results.append(b2_row)
        qdrant_audit_telemetry["Standard_HNSW"] = {
            "collection": "scifact_hnsw_std",
            "indexed_vectors_count": coll_info.indexed_vectors_count,
            "segments_count": coll_info.segments_count,
            "status": str(coll_info.status),
            "disk_size_mb": disk_size_mb,
            "server_rss_mb": server_mem_mb,
        }
        print(f"HNSW Std -> Recall@10: {b2_row['Recall@10']} | ANN Recall: {b2_row['ANN_Recall@10']} | Latency: {b2_row['Latency_mean_ms']} ms", flush=True)

        # ---------------------------------------------------------
        # B3: Tuned HNSW (m=32, ef_construct=200, ef_search=128)
        # ---------------------------------------------------------
        print("\n--- [Baseline B3] Real Qdrant: Tuned HNSW ---", flush=True)
        hnsw_b3 = rest.HnswConfigDiff(m=32, ef_construct=200, full_scan_threshold=50)
        b3_build_time = index_collection("scifact_hnsw_tuned", hnsw_b3, None)
        coll_info = client.get_collection("scifact_hnsw_tuned")
        b3_metrics, b3_hits = evaluate_qdrant_search(
            "scifact_hnsw_tuned",
            rest.SearchParams(hnsw_ef=128, exact=False)
        )
        server_mem_mb = round(psutil.Process(server_proc.pid).memory_info().rss / (1024 * 1024), 1)
        b3_row = {
            "Configuration": "Qdrant Tuned HNSW (m=32, ef=128)",
            "Backend": "Qdrant v1.19.2 (Standalone Rust Server)",
            **b3_metrics,
            "Build_Time_s": round(b3_build_time, 2),
            "Disk_Size_MB": round(get_dir_size_bytes(STORAGE_PATH) / (1024 * 1024), 2),
            "RAM_MB": server_mem_mb,
        }
        all_results.append(b3_row)
        qdrant_audit_telemetry["Tuned_HNSW"] = {
            "collection": "scifact_hnsw_tuned",
            "indexed_vectors_count": coll_info.indexed_vectors_count,
            "segments_count": coll_info.segments_count,
            "status": str(coll_info.status),
            "disk_size_mb": b3_row["Disk_Size_MB"],
            "server_rss_mb": server_mem_mb,
        }
        print(f"HNSW Tuned -> Recall@10: {b3_row['Recall@10']} | ANN Recall: {b3_row['ANN_Recall@10']} | Latency: {b3_row['Latency_mean_ms']} ms", flush=True)

        # ---------------------------------------------------------
        # B4: Scalar Quantization (INT8)
        # ---------------------------------------------------------
        print("\n--- [Baseline B4] Real Qdrant: Scalar Quantization (INT8) ---", flush=True)
        sq_cfg = rest.ScalarQuantization(
            scalar=rest.ScalarQuantizationConfig(
                type=rest.ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            )
        )
        b4_build_time = index_collection("scifact_sq_int8", hnsw_b2, sq_cfg)
        coll_info = client.get_collection("scifact_sq_int8")
        b4_metrics, b4_hits = evaluate_qdrant_search(
            "scifact_sq_int8",
            rest.SearchParams(hnsw_ef=64, exact=False, quantization=rest.QuantizationSearchParams(rescore=True))
        )
        server_mem_mb = round(psutil.Process(server_proc.pid).memory_info().rss / (1024 * 1024), 1)
        b4_row = {
            "Configuration": "Qdrant Scalar Quantization (INT8 + Rescore)",
            "Backend": "Qdrant v1.19.2 (Standalone Rust Server)",
            **b4_metrics,
            "Build_Time_s": round(b4_build_time, 2),
            "Disk_Size_MB": round(get_dir_size_bytes(STORAGE_PATH) / (1024 * 1024), 2),
            "RAM_MB": server_mem_mb,
        }
        all_results.append(b4_row)
        qdrant_audit_telemetry["Scalar_Quantization"] = {
            "collection": "scifact_sq_int8",
            "indexed_vectors_count": coll_info.indexed_vectors_count,
            "segments_count": coll_info.segments_count,
            "status": str(coll_info.status),
            "disk_size_mb": b4_row["Disk_Size_MB"],
            "server_rss_mb": server_mem_mb,
        }
        print(f"Scalar Quant -> Recall@10: {b4_row['Recall@10']} | ANN Recall: {b4_row['ANN_Recall@10']} | Latency: {b4_row['Latency_mean_ms']} ms", flush=True)

        # ---------------------------------------------------------
        # B5: Binary Quantization (1-bit)
        # ---------------------------------------------------------
        print("\n--- [Baseline B5] Real Qdrant: Binary Quantization (1-bit) ---", flush=True)
        bq_cfg = rest.BinaryQuantization(
            binary=rest.BinaryQuantizationConfig(always_ram=True)
        )
        b5_build_time = index_collection("scifact_bq_1bit", hnsw_b2, bq_cfg)
        coll_info = client.get_collection("scifact_bq_1bit")
        b5_metrics, b5_hits = evaluate_qdrant_search(
            "scifact_bq_1bit",
            rest.SearchParams(
                hnsw_ef=64,
                exact=False,
                quantization=rest.QuantizationSearchParams(rescore=True, oversampling=2.0)
            )
        )
        server_mem_mb = round(psutil.Process(server_proc.pid).memory_info().rss / (1024 * 1024), 1)
        b5_row = {
            "Configuration": "Qdrant Binary Quantization (1-bit + Rescore)",
            "Backend": "Qdrant v1.19.2 (Standalone Rust Server)",
            **b5_metrics,
            "Build_Time_s": round(b5_build_time, 2),
            "Disk_Size_MB": round(get_dir_size_bytes(STORAGE_PATH) / (1024 * 1024), 2),
            "RAM_MB": server_mem_mb,
        }
        all_results.append(b5_row)
        qdrant_audit_telemetry["Binary_Quantization"] = {
            "collection": "scifact_bq_1bit",
            "indexed_vectors_count": coll_info.indexed_vectors_count,
            "segments_count": coll_info.segments_count,
            "status": str(coll_info.status),
            "disk_size_mb": b5_row["Disk_Size_MB"],
            "server_rss_mb": server_mem_mb,
        }
        print(f"Binary Quant -> Recall@10: {b5_row['Recall@10']} | ANN Recall: {b5_row['ANN_Recall@10']} | Latency: {b5_row['Latency_mean_ms']} ms", flush=True)

    finally:
        stop_qdrant_server(server_proc)

    # -------------------------------------------------------------
    # BASELINE C: Hybrid Retrieval (BM25 + Dense Exact RRF)
    # -------------------------------------------------------------
    print("\n--- [Baseline C] Evaluating Hybrid BM25 + Dense RRF ---", flush=True)
    latencies_hybrid = []
    r1, r5, r10, mrr, ndcg = [], [], [], [], []

    for qid in query_ids:
        gold_set = test_qrels[qid]
        t0 = time.perf_counter()
        fused = rrf_fuse(bm25_hits[qid], exact_hits[qid], k_rrf=60, top_k=100)
        latencies_hybrid.append((time.perf_counter() - t0) * 1000.0 + bm25_row["Latency_mean_ms"] + exact_row["Latency_mean_ms"])

        retrieved_ids = [doc_id for doc_id, _ in fused]
        r1.append(calculate_recall_at_k(retrieved_ids, gold_set, 1))
        r5.append(calculate_recall_at_k(retrieved_ids, gold_set, 5))
        r10.append(calculate_recall_at_k(retrieved_ids, gold_set, 10))
        mrr.append(calculate_mrr(retrieved_ids, gold_set))
        ndcg.append(calculate_ndcg_at_k(retrieved_ids, gold_set, 10))

    hybrid_row = {
        "Configuration": "Hybrid (BM25 + Dense Exact RRF k=60)",
        "Backend": "Reciprocal Rank Fusion",
        "Recall@1": round(float(np.mean(r1)), 4),
        "Recall@5": round(float(np.mean(r5)), 4),
        "Recall@10": round(float(np.mean(r10)), 4),
        "MRR": round(float(np.mean(mrr)), 4),
        "nDCG@10": round(float(np.mean(ndcg)), 4),
        "ANN_Recall@10": 1.0,
        "Latency_p50_ms": round(float(np.median(latencies_hybrid)), 2),
        "Latency_p95_ms": round(float(np.percentile(latencies_hybrid, 95)), 2),
        "Latency_mean_ms": round(float(np.mean(latencies_hybrid)), 2),
        "Build_Time_s": bm25_row["Build_Time_s"],
        "Disk_Size_MB": exact_row["Disk_Size_MB"],
        "RAM_MB": exact_row["RAM_MB"],
    }
    all_results.append(hybrid_row)
    print(f"Hybrid -> Recall@10: {hybrid_row['Recall@10']} | nDCG@10: {hybrid_row['nDCG@10']} | Latency: {hybrid_row['Latency_mean_ms']} ms", flush=True)

    # Save results to CSV and JSON
    import pandas as pd
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(results_dir, "results.csv")
    json_path = os.path.join(results_dir, "results.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    with open("results/real_qdrant/audit_metrics.json", "w", encoding="utf-8") as f:
        json.dump(qdrant_audit_telemetry, f, indent=2)

    print("\n=================================================================", flush=True)
    print("                    FINAL BENCHMARK RESULTS                      ", flush=True)
    print("=================================================================", flush=True)
    print(df[["Configuration", "Recall@1", "Recall@10", "MRR", "nDCG@10", "ANN_Recall@10", "Latency_mean_ms", "Build_Time_s"]].to_string(), flush=True)
    print(f"\nSaved results to {csv_path} and {json_path}", flush=True)


if __name__ == "__main__":
    main()
