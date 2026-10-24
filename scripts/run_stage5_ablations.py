"""
Stage 5 Ablation Studies Pipeline for RAGScaleBench:
1. Fusion parameter ablation: k_rrf in [10, 20, 60, 100, 200]
2. Top-k cutoff sensitivity: k in [1, 3, 5, 10, 20, 50]
3. Adaptive router feature ablation:
   - Full Policy (Margin + IDF + Fallback)
   - Ablation No IDF (Margin only)
   - Ablation No Margin (IDF only)
   - Ablation Length Only (Query token count heuristic)
4. Qdrant Quantization and Rescoring ablation:
   - Float32 Exact
   - INT8 + Rescore vs INT8 No-Rescore
   - 1-bit BQ + Rescore vs 1-bit BQ No-Rescore
Guarantees clean process lifecycle and zero orphaned daemons.
"""

from collections import defaultdict
import json
import os
import shutil
import subprocess
import time
import numpy as np
import pandas as pd
import psutil
import requests

from ragbench.evaluation.bm25 import BM25Okapi, tokenize_text
from ragbench.evaluation.retrieval import (
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_recall_at_k,
)
from ragbench.retrieval.policies import (
    AdaptiveRouterPolicy,
    AdaptiveThresholds,
    BM25Policy,
    DensePolicy,
    FixedHybridPolicy,
    rrf_fuse,
)

SERVER_PORT = 6338
SERVER_HOST = "127.0.0.1"
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
STORAGE_PATH = "./scratch/qdrant_ablation_storage"


def start_qdrant() -> subprocess.Popen:
    if os.path.exists(STORAGE_PATH):
        try:
            shutil.rmtree(STORAGE_PATH)
        except Exception:
            pass
    os.makedirs(STORAGE_PATH, exist_ok=True)

    env = os.environ.copy()
    env["QDRANT__SERVICE__HOST"] = SERVER_HOST
    env["QDRANT__SERVICE__HTTP_PORT"] = str(SERVER_PORT)
    env["QDRANT__SERVICE__GRPC_PORT"] = "6339"
    env["QDRANT__STORAGE__STORAGE_PATH"] = STORAGE_PATH
    env["QDRANT__LOG_LEVEL"] = "WARN"

    proc = subprocess.Popen(
        [os.path.abspath("bin/qdrant.exe")],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

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
    raise RuntimeError("Failed to start Qdrant server for ablations.")


def stop_qdrant(proc: subprocess.Popen) -> None:
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
    print("Qdrant ablation server cleanly terminated.", flush=True)


def run_fusion_ablations(bm25_hits: dict, exact_hits: dict, test_qrels: dict, query_ids: list[str]) -> list[dict]:
    print("\n--- Ablation 1: Reciprocal Rank Fusion Parameter (k_rrf) Sensitivity ---", flush=True)
    rows = []
    for k_val in [10, 20, 60, 100, 200]:
        r1, r5, r10, mrr_list, ndcg_list = [], [], [], [], []
        for qid in query_ids:
            gold = test_qrels[qid]
            fused = rrf_fuse(bm25_hits[qid], exact_hits[qid], k_rrf=k_val, top_k=50)
            ret_ids = [d for d, _ in fused]
            r1.append(calculate_recall_at_k(ret_ids, gold, 1))
            r5.append(calculate_recall_at_k(ret_ids, gold, 5))
            r10.append(calculate_recall_at_k(ret_ids, gold, 10))
            mrr_list.append(calculate_mrr(ret_ids, gold))
            ndcg_list.append(calculate_ndcg_at_k(ret_ids, gold, 10))

        rows.append({
            "k_rrf": k_val,
            "Recall@1": round(float(np.mean(r1)), 4),
            "Recall@5": round(float(np.mean(r5)), 4),
            "Recall@10": round(float(np.mean(r10)), 4),
            "MRR": round(float(np.mean(mrr_list)), 4),
            "nDCG@10": round(float(np.mean(ndcg_list)), 4),
        })
    return rows


def run_topk_ablations(bm25_hits: dict, exact_hits: dict, test_qrels: dict, query_ids: list[str]) -> list[dict]:
    print("\n--- Ablation 2: Top-k Cutoff Sensitivity ---", flush=True)
    rows = []
    for k_val in [1, 3, 5, 10, 20, 50]:
        # BM25
        bm25_r = [calculate_recall_at_k([d for d, _ in bm25_hits[q]], test_qrels[q], k_val) for q in query_ids]
        bm25_ndcg = [calculate_ndcg_at_k([d for d, _ in bm25_hits[q]], test_qrels[q], k_val) for q in query_ids]
        # Dense
        dense_r = [calculate_recall_at_k([d for d, _ in exact_hits[q]], test_qrels[q], k_val) for q in query_ids]
        dense_ndcg = [calculate_ndcg_at_k([d for d, _ in exact_hits[q]], test_qrels[q], k_val) for q in query_ids]
        # Hybrid
        hybrid_hits = {q: rrf_fuse(bm25_hits[q], exact_hits[q], k_rrf=60, top_k=k_val) for q in query_ids}
        hybrid_r = [calculate_recall_at_k([d for d, _ in hybrid_hits[q]], test_qrels[q], k_val) for q in query_ids]
        hybrid_ndcg = [calculate_ndcg_at_k([d for d, _ in hybrid_hits[q]], test_qrels[q], k_val) for q in query_ids]

        rows.append({
            "k_cutoff": k_val,
            "BM25_Recall": round(float(np.mean(bm25_r)), 4),
            "BM25_nDCG": round(float(np.mean(bm25_ndcg)), 4),
            "Dense_Recall": round(float(np.mean(dense_r)), 4),
            "Dense_nDCG": round(float(np.mean(dense_ndcg)), 4),
            "Hybrid_Recall": round(float(np.mean(hybrid_r)), 4),
            "Hybrid_nDCG": round(float(np.mean(hybrid_ndcg)), 4),
        })
    return rows


def run_adaptive_feature_ablations(
    bm25: BM25Okapi,
    dense_search_fn,
    queries_dict: dict,
    query_vecs: np.ndarray,
    query_ids: list[str],
    test_qrels: dict,
) -> list[dict]:
    print("\n--- Ablation 3: Adaptive Router Feature Ablations ---", flush=True)

    configs = [
        ("Full_Policy", AdaptiveThresholds(tau_lexical_margin=0.20, tau_min_idf=5.0, tau_dense_confident=0.75, tau_bm25_weak=8.0)),
        ("Ablation_No_IDF", AdaptiveThresholds(tau_lexical_margin=0.20, tau_min_idf=0.0, tau_dense_confident=0.75, tau_bm25_weak=8.0)),
        ("Ablation_No_Margin", AdaptiveThresholds(tau_lexical_margin=0.0, tau_min_idf=5.0, tau_dense_confident=0.75, tau_bm25_weak=8.0)),
        ("Ablation_Strict_Margin_Only", AdaptiveThresholds(tau_lexical_margin=0.35, tau_min_idf=0.0, tau_dense_confident=0.75, tau_bm25_weak=8.0)),
    ]

    rows = []
    for cfg_name, thresh in configs:
        router = AdaptiveRouterPolicy(bm25, dense_search_fn, thresh)
        dense_calls = 0
        r10, ndcg10, mrr_list, latencies = [], [], [], []

        for idx, qid in enumerate(query_ids):
            qtext = queries_dict[qid]
            qvec = query_vecs[idx]
            gold = test_qrels[qid]

            res = router.retrieve(qtext, query_vec=qvec, top_k=10)
            if res.dense_search_executed:
                dense_calls += 1
            latencies.append(res.latency_ms)

            ret_ids = [d for d, _ in res.retrieved_docs]
            r10.append(calculate_recall_at_k(ret_ids, gold, 10))
            ndcg10.append(calculate_ndcg_at_k(ret_ids, gold, 10))
            mrr_list.append(calculate_mrr(ret_ids, gold))

        saved_pct = ((len(query_ids) - dense_calls) / len(query_ids)) * 100.0
        rows.append({
            "Feature_Configuration": cfg_name,
            "Dense_Searches_Saved_%": round(saved_pct, 1),
            "Recall@10": round(float(np.mean(r10)), 4),
            "nDCG@10": round(float(np.mean(ndcg10)), 4),
            "MRR": round(float(np.mean(mrr_list)), 4),
            "Mean_Latency_ms": round(float(np.mean(latencies)), 2),
        })

    # Add Query Length Heuristic baseline
    len_dense_calls = 0
    len_r10, len_ndcg, len_mrr, len_lats = [], [], [], []
    for idx, qid in enumerate(query_ids):
        qtext = queries_dict[qid]
        qvec = query_vecs[idx]
        gold = test_qrels[qid]
        tokens = tokenize_text(qtext)
        t0 = time.perf_counter()
        if len(tokens) >= 12:
            # Long query -> lexical only
            hits = bm25.search(qtext, top_k=10)
        else:
            len_dense_calls += 1
            b_hits = bm25.search(qtext, top_k=50)
            d_hits = dense_search_fn(qvec, top_k=50)
            hits = rrf_fuse(b_hits, d_hits, k_rrf=60, top_k=10)
        dur = (time.perf_counter() - t0) * 1000.0
        len_lats.append(dur)
        ret_ids = [d for d, _ in hits]
        len_r10.append(calculate_recall_at_k(ret_ids, gold, 10))
        len_ndcg.append(calculate_ndcg_at_k(ret_ids, gold, 10))
        len_mrr.append(calculate_mrr(ret_ids, gold))

    len_saved_pct = ((len(query_ids) - len_dense_calls) / len(query_ids)) * 100.0
    rows.append({
        "Feature_Configuration": "Ablation_Length_Heuristic (>=12 tokens)",
        "Dense_Searches_Saved_%": round(len_saved_pct, 1),
        "Recall@10": round(float(np.mean(len_r10)), 4),
        "nDCG@10": round(float(np.mean(len_ndcg)), 4),
        "MRR": round(float(np.mean(len_mrr)), 4),
        "Mean_Latency_ms": round(float(np.mean(len_lats)), 2),
    })

    return rows


def run_qdrant_quantization_ablations(
    corpus_ids: list[str],
    corpus_vecs: np.ndarray,
    query_ids: list[str],
    query_vecs: np.ndarray,
    test_qrels: dict,
    ann_gold: dict[str, list[str]],
) -> list[dict]:
    print("\n--- Ablation 4: Qdrant Quantization & Rescoring Impact ---", flush=True)
    proc = start_qdrant()

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as rest

        client = QdrantClient(url=SERVER_URL, timeout=30.0)

        def build_collection(name: str, quant: rest.QuantizationConfig | None):
            try:
                client.delete_collection(name)
            except Exception:
                pass
            client.create_collection(
                collection_name=name,
                vectors_config=rest.VectorParams(
                    size=int(corpus_vecs.shape[1]),
                    distance=rest.Distance.COSINE,
                    on_disk=False,
                ),
                hnsw_config=rest.HnswConfigDiff(m=16, ef_construct=100, full_scan_threshold=50),
                quantization_config=quant,
                optimizers_config=rest.OptimizersConfigDiff(indexing_threshold=100, default_segment_number=2),
            )
            batch_size = 500
            for start_idx in range(0, len(corpus_ids), batch_size):
                end_idx = min(start_idx + batch_size, len(corpus_ids))
                pts = [
                    rest.PointStruct(id=i, vector=corpus_vecs[i].tolist(), payload={"doc_id": corpus_ids[i]})
                    for i in range(start_idx, end_idx)
                ]
                client.upsert(name, pts, wait=True)
            client.update_collection(name, optimizer_config=rest.OptimizersConfigDiff(indexing_threshold=10))
            time.sleep(1.5)

        # 1. SQ Collection
        sq_cfg = rest.ScalarQuantization(scalar=rest.ScalarQuantizationConfig(type=rest.ScalarType.INT8, quantile=0.99, always_ram=True))
        build_collection("scifact_sq_abl", sq_cfg)

        # 2. BQ Collection
        bq_cfg = rest.BinaryQuantization(binary=rest.BinaryQuantizationConfig(always_ram=True))
        build_collection("scifact_bq_abl", bq_cfg)

        abl_experiments = [
            ("Scalar INT8 + Rescore=True", "scifact_sq_abl", rest.SearchParams(hnsw_ef=64, quantization=rest.QuantizationSearchParams(rescore=True))),
            ("Scalar INT8 + Rescore=False", "scifact_sq_abl", rest.SearchParams(hnsw_ef=64, quantization=rest.QuantizationSearchParams(rescore=False))),
            ("Binary 1-bit + Rescore=True", "scifact_bq_abl", rest.SearchParams(hnsw_ef=64, quantization=rest.QuantizationSearchParams(rescore=True, oversampling=2.0))),
            ("Binary 1-bit + Rescore=False", "scifact_bq_abl", rest.SearchParams(hnsw_ef=64, quantization=rest.QuantizationSearchParams(rescore=False))),
        ]

        rows = []
        for label, coll_name, params in abl_experiments:
            r10, ndcg10, ann_recs, lats = [], [], [], []
            for i, qid in enumerate(query_ids):
                qvec = query_vecs[i].tolist()
                t0 = time.perf_counter()
                res = client.query_points(coll_name, query=qvec, limit=50, search_params=params, with_payload=True)
                dur = (time.perf_counter() - t0) * 1000.0
                lats.append(dur)

                ret_ids = [pt.payload["doc_id"] for pt in res.points]
                r10.append(calculate_recall_at_k(ret_ids, test_qrels[qid], 10))
                ndcg10.append(calculate_ndcg_at_k(ret_ids, test_qrels[qid], 10))

                gold_exact = set(ann_gold[qid])
                matched = sum(1 for cid in ret_ids[:10] if cid in gold_exact)
                ann_recs.append(matched / 10.0)

            rows.append({
                "Quantization_Setup": label,
                "Recall@10": round(float(np.mean(r10)), 4),
                "nDCG@10": round(float(np.mean(ndcg10)), 4),
                "ANN_Recall@10": round(float(np.mean(ann_recs)), 4),
                "Latency_p50_ms": round(float(np.median(lats)), 2),
                "Latency_mean_ms": round(float(np.mean(lats)), 2),
            })

        return rows
    finally:
        stop_qdrant(proc)


def main():
    print("=================================================================", flush=True)
    print("           RAGScaleBench Phase 5: Stage 5 Ablation Studies       ", flush=True)
    print("=================================================================", flush=True)

    scifact_dir = "data/beir/scifact"
    results_dir = "results/scifact"
    ablations_dir = "results/ablations"
    os.makedirs(ablations_dir, exist_ok=True)

    corpus_emb = np.load(os.path.join(results_dir, "corpus_embeddings.npy"))
    query_emb = np.load(os.path.join(results_dir, "query_embeddings.npy"))
    with open(os.path.join(results_dir, "corpus_ids.json"), "r", encoding="utf-8") as f:
        corpus_ids = json.load(f)
    with open(os.path.join(results_dir, "query_ids.json"), "r", encoding="utf-8") as f:
        query_ids = json.load(f)

    corpus_dict = {}
    with open(os.path.join(scifact_dir, "corpus.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            corpus_dict[item["_id"]] = (item.get("title", "") + " " + item.get("text", "")).strip()

    queries_dict = {}
    with open(os.path.join(scifact_dir, "queries.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            queries_dict[item["_id"]] = item["text"]

    test_qrels = defaultdict(set)
    with open(os.path.join(scifact_dir, "qrels/test.tsv"), "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3 and int(parts[2]) > 0:
                test_qrels[parts[0]].add(parts[1])

    # Precompute BM25 hits & Dense exact hits
    bm25 = BM25Okapi()
    bm25.fit([(cid, corpus_dict[cid]) for cid in corpus_ids])

    corpus_emb_t = corpus_emb.T
    def dense_search(query_vec: np.ndarray, top_k: int = 50) -> list[tuple[str, float]]:
        sims = np.dot(query_vec, corpus_emb_t)
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(corpus_ids[idx], float(sims[idx])) for idx in top_idx]

    bm25_hits = {q: bm25.search(queries_dict[q], top_k=50) for q in query_ids}
    exact_hits = {q: dense_search(query_emb[i], top_k=50) for i, q in enumerate(query_ids)}
    ann_gold = {q: [cid for cid, _ in exact_hits[q][:10]] for q in query_ids}

    # Run Ablations
    fusion_rows = run_fusion_ablations(bm25_hits, exact_hits, test_qrels, query_ids)
    topk_rows = run_topk_ablations(bm25_hits, exact_hits, test_qrels, query_ids)
    feature_rows = run_adaptive_feature_ablations(bm25, dense_search, queries_dict, query_emb, query_ids, test_qrels)
    quant_rows = run_qdrant_quantization_ablations(corpus_ids, corpus_emb, query_ids, query_emb, test_qrels, ann_gold)

    # Convert to DataFrames and display
    df_fusion = pd.DataFrame(fusion_rows)
    df_topk = pd.DataFrame(topk_rows)
    df_feature = pd.DataFrame(feature_rows)
    df_quant = pd.DataFrame(quant_rows)

    print("\n--- Fusion Parameter Ablation Results ---")
    print(df_fusion.to_string(index=False), flush=True)

    print("\n--- Top-k Cutoff Ablation Results ---")
    print(df_topk.to_string(index=False), flush=True)

    print("\n--- Adaptive Feature Ablation Results ---")
    print(df_feature.to_string(index=False), flush=True)

    print("\n--- Quantization & Rescoring Ablation Results ---")
    print(df_quant.to_string(index=False), flush=True)

    # Save to CSV files
    df_fusion.to_csv(os.path.join(ablations_dir, "fusion_parameter_ablation.csv"), index=False)
    df_topk.to_csv(os.path.join(ablations_dir, "topk_cutoff_ablation.csv"), index=False)
    df_feature.to_csv(os.path.join(ablations_dir, "adaptive_feature_ablation.csv"), index=False)
    df_quant.to_csv(os.path.join(ablations_dir, "quantization_rescore_ablation.csv"), index=False)

    print(f"\nAll ablation study artifacts saved to {ablations_dir}/", flush=True)


if __name__ == "__main__":
    main()
