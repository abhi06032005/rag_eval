"""
BEIR FiQA Embedding, Adaptive Retrieval, and Statistical Validation Pipeline.
Operates with capped thread usage (torch.set_num_threads(3)) to protect local hardware.
Includes resumable chunked embedding caching, progress tracking, and full policy comparison:
  A. BM25 (Lexical)
  B. Dense Exact (Brute Force Cosine)
  C. Fixed Hybrid RRF (k=60)
  D. Adaptive Quality-Aware Router
Performs paired query-level bootstrap evaluation (B=10,000 resamples) and saves all artifacts.
"""

from collections import defaultdict
import glob
import json
import os
import time
import numpy as np
import pandas as pd
import psutil
import torch

from ragbench.evaluation.bm25 import BM25Okapi, tokenize_text
from ragbench.evaluation.bootstrap import paired_query_bootstrap
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
)

# Cap PyTorch CPU threads to protect user hardware from thermal stress
torch.set_num_threads(3)
torch.set_num_interop_threads(1)


def embed_and_cache_fiqa():
    fiqa_dir = "data/beir/fiqa"
    results_dir = "results/fiqa"
    os.makedirs(results_dir, exist_ok=True)
    temp_dir = os.path.join(results_dir, "shards")
    os.makedirs(temp_dir, exist_ok=True)

    corpus_emb_path = os.path.join(results_dir, "corpus_embeddings.npy")
    query_emb_path = os.path.join(results_dir, "query_embeddings.npy")
    corpus_ids_path = os.path.join(results_dir, "corpus_ids.json")
    query_ids_path = os.path.join(results_dir, "query_ids.json")

    # 1. Load Corpus
    print("Loading FiQA corpus...", flush=True)
    corpus_ids = []
    corpus_texts = []
    corpus_dict = {}
    with open(os.path.join(fiqa_dir, "corpus.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            cid = str(item["_id"])
            text = (item.get("title", "") + " " + item.get("text", "")).strip()
            corpus_ids.append(cid)
            corpus_texts.append(text)
            corpus_dict[cid] = text

    print(f"Loaded {len(corpus_ids)} documents from FiQA corpus.", flush=True)

    # 2. Load Queries and Test Qrels
    test_qrels = defaultdict(set)
    with open(os.path.join(fiqa_dir, "qrels/test.tsv"), "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3 and int(parts[2]) > 0:
                test_qrels[str(parts[0])].add(str(parts[1]))

    queries_dict = {}
    with open(os.path.join(fiqa_dir, "queries.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            queries_dict[str(item["_id"])] = item["text"]

    test_qids = [qid for qid in test_qrels if qid in queries_dict]
    test_qtexts = [queries_dict[qid] for qid in test_qids]
    print(f"Loaded {len(test_qids)} test queries with positive test qrels.", flush=True)

    # 3. Load Dev Qrels for Zero-Leakage Policy Calibration
    dev_qrels = defaultdict(set)
    with open(os.path.join(fiqa_dir, "qrels/dev.tsv"), "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3 and int(parts[2]) > 0:
                dev_qrels[str(parts[0])].add(str(parts[1]))

    # Save ID lists
    with open(corpus_ids_path, "w", encoding="utf-8") as f:
        json.dump(corpus_ids, f)
    with open(query_ids_path, "w", encoding="utf-8") as f:
        json.dump(test_qids, f)

    # 4. Generate or Load Dense Embeddings
    from sentence_transformers import SentenceTransformer
    model = None

    if not os.path.exists(query_emb_path):
        print(f"Generating query embeddings for {len(test_qids)} FiQA test queries...", flush=True)
        if model is None:
            model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        query_vecs = model.encode(
            test_qtexts,
            batch_size=128,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        np.save(query_emb_path, query_vecs)
        print(f"Saved query embeddings to {query_emb_path} (Shape: {query_vecs.shape})", flush=True)
    else:
        query_vecs = np.load(query_emb_path)
        print(f"Loaded existing query embeddings from {query_emb_path}", flush=True)

    if not os.path.exists(corpus_emb_path):
        print(f"Generating dense embeddings for {len(corpus_ids)} FiQA documents...", flush=True)
        if model is None:
            model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        shard_size = 5000
        n_shards = (len(corpus_texts) + shard_size - 1) // shard_size
        shard_files = []

        t_start_all = time.perf_counter()
        for s_idx in range(n_shards):
            s_path = os.path.join(temp_dir, f"fiqa_shard_{s_idx:03d}.npy")
            shard_files.append(s_path)
            if os.path.exists(s_path):
                print(f"Shard {s_idx+1}/{n_shards} already exists, skipping...", flush=True)
                continue

            start_i = s_idx * shard_size
            end_i = min(len(corpus_texts), (s_idx + 1) * shard_size)
            chunk_texts = [t[:512] for t in corpus_texts[start_i:end_i]]

            t_shard = time.perf_counter()
            shard_vecs = model.encode(
                chunk_texts,
                batch_size=256,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            np.save(s_path, shard_vecs)
            dur_shard = time.perf_counter() - t_shard
            rate = len(chunk_texts) / max(dur_shard, 0.001)
            print(f"Completed shard {s_idx+1}/{n_shards} ({len(chunk_texts)} docs in {dur_shard:.1f}s, {rate:.1f} docs/s)", flush=True)

        # Concatenate shards
        print("Concatenating all shards into unified corpus_embeddings.npy...", flush=True)
        all_vecs = [np.load(p) for p in shard_files]
        corpus_vecs = np.vstack(all_vecs)
        np.save(corpus_emb_path, corpus_vecs)
        total_time = time.perf_counter() - t_start_all
        print(f"Successfully saved {corpus_emb_path}. Total shape: {corpus_vecs.shape} in {total_time:.1f}s", flush=True)

        # Clean up shards
        for p in shard_files:
            try:
                os.remove(p)
            except Exception:
                pass
        try:
            os.rmdir(temp_dir)
        except Exception:
            pass
    else:
        corpus_vecs = np.load(corpus_emb_path)
        print(f"Loaded existing corpus embeddings from {corpus_emb_path} (Shape: {corpus_vecs.shape})", flush=True)

    return {
        "corpus_ids": corpus_ids,
        "corpus_vecs": corpus_vecs,
        "corpus_dict": corpus_dict,
        "test_query_ids": test_qids,
        "test_query_vecs": query_vecs,
        "queries_dict": queries_dict,
        "test_qrels": test_qrels,
        "dev_qrels": dev_qrels,
    }


def calibrate_fiqa_dev(bm25: BM25Okapi, queries_dict: dict, dev_qrels: dict) -> AdaptiveThresholds:
    """Calibrate adaptive thresholds strictly on FiQA dev set without touching test set."""
    print("\n--- Calibrating Adaptive Router on Independent FiQA Dev Set ---", flush=True)
    dev_qids = [qid for qid in dev_qrels if qid in queries_dict]
    print(f"Calibrating across {len(dev_qids)} dev queries...", flush=True)

    records = []
    for qid in dev_qids[:300]:
        qtext = queries_dict[qid]
        gold_set = dev_qrels[qid]
        tokens = tokenize_text(qtext)
        idfs = [bm25.idf.get(t, 0.0) for t in tokens]
        max_idf = max(idfs) if idfs else 0.0
        hits = bm25.search(qtext, top_k=10)
        if not hits:
            continue
        top_score = hits[0][1]
        runner_up = hits[1][1] if len(hits) > 1 else 0.0
        margin = (top_score - runner_up) / (top_score + 1e-6)
        top1_correct = hits[0][0] in gold_set
        records.append({
            "qid": qid,
            "margin": margin,
            "max_idf": max_idf,
            "top1_correct": top1_correct,
        })

    best_tau_margin = 0.25
    best_tau_idf = 5.0
    best_score = -1.0

    for m_cand in [0.15, 0.20, 0.25, 0.30]:
        for idf_cand in [4.0, 5.0, 6.0]:
            subset = [r for r in records if r["margin"] >= m_cand and r["max_idf"] >= idf_cand]
            if len(subset) >= 20:
                acc = np.mean([r["top1_correct"] for r in subset])
                cov = len(subset) / len(records)
                score = acc * np.sqrt(cov)
                if score > best_score:
                    best_score = score
                    best_tau_margin = m_cand
                    best_tau_idf = idf_cand

    print(f"FiQA Calibration Result: tau_lexical_margin={best_tau_margin}, tau_min_idf={best_tau_idf}", flush=True)
    return AdaptiveThresholds(
        tau_lexical_margin=best_tau_margin,
        tau_min_idf=best_tau_idf,
        tau_dense_confident=0.75,
        tau_bm25_weak=8.0,
        k_rrf=60,
    )


def evaluate_fiqa_benchmark():
    t_start = time.perf_counter()
    data = embed_and_cache_fiqa()
    corpus_ids = data["corpus_ids"]
    corpus_vecs = data["corpus_vecs"]
    corpus_dict = data["corpus_dict"]
    test_qids = data["test_query_ids"]
    test_qvecs = data["test_query_vecs"]
    queries_dict = data["queries_dict"]
    test_qrels = data["test_qrels"]
    dev_qrels = data["dev_qrels"]

    # 1. Build BM25 Okapi index
    print(f"\nBuilding BM25 Okapi index for {len(corpus_ids)} FiQA documents...", flush=True)
    t0_bm25 = time.perf_counter()
    bm25 = BM25Okapi()
    bm25.fit([(cid, corpus_dict[cid]) for cid in corpus_ids])
    bm25_build_time = time.perf_counter() - t0_bm25
    print(f"BM25 index built in {bm25_build_time:.2f} seconds.", flush=True)

    # 2. Dense search function (NumPy exact)
    corpus_vecs_t = corpus_vecs.T
    def dense_search(query_vec: np.ndarray, top_k: int = 50) -> list[tuple[str, float]]:
        sims = np.dot(query_vec, corpus_vecs_t)
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(corpus_ids[idx], float(sims[idx])) for idx in top_idx]

    # 3. Zero-leakage dev calibration
    calibrated_thresholds = calibrate_fiqa_dev(bm25, queries_dict, dev_qrels)

    # 4. Initialize Policies
    policy_bm25 = BM25Policy(bm25)
    policy_dense = DensePolicy(dense_search)
    policy_hybrid = FixedHybridPolicy(bm25, dense_search, k_rrf=60)
    policy_adaptive = AdaptiveRouterPolicy(bm25, dense_search, calibrated_thresholds)

    policies = {
        "BM25": policy_bm25,
        "Dense_Exact": policy_dense,
        "Fixed_Hybrid": policy_hybrid,
        "Adaptive_Router": policy_adaptive,
    }

    # 5. Evaluate all policies across locked test queries
    print(f"\n--- Evaluating Policies on {len(test_qids)} Locked FiQA Test Queries ---", flush=True)
    per_query_metrics = {name: {"recall10": {}, "ndcg10": {}, "mrr": {}} for name in policies}
    routing_records = []
    dense_searches_executed = 0
    policy_latencies = {name: [] for name in policies}

    for idx, qid in enumerate(test_qids):
        qtext = queries_dict[qid]
        qvec = test_qvecs[idx]
        gold_set = test_qrels[qid]

        for name, policy in policies.items():
            t_p = time.perf_counter()
            res = policy.retrieve(qtext, query_vec=qvec, top_k=50)
            policy_latencies[name].append((time.perf_counter() - t_p) * 1000.0)

            retrieved_ids = [doc_id for doc_id, _ in res.retrieved_docs]
            r10 = calculate_recall_at_k(retrieved_ids, gold_set, 10)
            ndcg10 = calculate_ndcg_at_k(retrieved_ids, gold_set, 10)
            mrr_val = calculate_mrr(retrieved_ids, gold_set)

            per_query_metrics[name]["recall10"][qid] = r10
            per_query_metrics[name]["ndcg10"][qid] = ndcg10
            per_query_metrics[name]["mrr"][qid] = mrr_val

            if name == "Adaptive_Router":
                if res.dense_search_executed:
                    dense_searches_executed += 1
                routing_records.append({
                    "query_id": qid,
                    "query_text": qtext,
                    "route_taken": res.route_taken,
                    "reason": res.reason,
                    "dense_executed": res.dense_search_executed,
                    "latency_ms": round(res.latency_ms, 2),
                    "recall10": round(r10, 4),
                    "ndcg10": round(ndcg10, 4),
                })

    # Summary table
    process_rss = round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)
    disk_size_mb = round(os.path.getsize(os.path.join("results/fiqa", "corpus_embeddings.npy")) / (1024 * 1024), 2)

    summary_rows = []
    for name in policies:
        r10_mean = float(np.mean(list(per_query_metrics[name]["recall10"].values())))
        ndcg10_mean = float(np.mean(list(per_query_metrics[name]["ndcg10"].values())))
        mrr_mean = float(np.mean(list(per_query_metrics[name]["mrr"].values())))
        p50_lat = float(np.median(policy_latencies[name]))
        mean_lat = float(np.mean(policy_latencies[name]))
        backend_name = (
            "Pure Python Inverted Index" if name == "BM25"
            else "NumPy Exact Cosine Matrix Product" if name == "Dense_Exact"
            else "Reciprocal Rank Fusion (k=60)" if name == "Fixed_Hybrid"
            else "Query-Adaptive Quality-Aware Router"
        )
        summary_rows.append({
            "Configuration": name,
            "Backend": backend_name,
            "Recall@10": round(r10_mean, 4),
            "nDCG@10": round(ndcg10_mean, 4),
            "MRR": round(mrr_mean, 4),
            "Latency_p50_ms": round(p50_lat, 2),
            "Latency_mean_ms": round(mean_lat, 2),
            "Build_Time_s": round(bm25_build_time, 2),
            "Index_Disk_MB": disk_size_mb if name != "BM25" else 0.0,
            "Process_RAM_MB": process_rss,
        })

    df_summary = pd.DataFrame(summary_rows)
    print("\nFiQA Summary Evaluation Metrics:")
    print(df_summary.to_string(index=False), flush=True)

    # Dense savings
    total_q = len(test_qids)
    saved_dense = total_q - dense_searches_executed
    saved_pct = (saved_dense / total_q) * 100.0
    print(f"\nFiQA Adaptive Routing: Saved {saved_dense}/{total_q} ({saved_pct:.1f}%) dense searches.", flush=True)

    # 6. Paired Query Bootstrap
    print("\n--- Running Paired Query Bootstrap for FiQA (B=10,000) ---", flush=True)
    comparisons = [
        ("BM25", "Dense_Exact"),
        ("Dense_Exact", "Fixed_Hybrid"),
        ("BM25", "Fixed_Hybrid"),
        ("Fixed_Hybrid", "Adaptive_Router"),
        ("BM25", "Adaptive_Router"),
        ("Dense_Exact", "Adaptive_Router"),
    ]

    bootstrap_results = {}
    bootstrap_table = []

    for name_a, name_b in comparisons:
        for metric in ["ndcg10", "recall10"]:
            scores_a = per_query_metrics[name_a][metric]
            scores_b = per_query_metrics[name_b][metric]
            m_label = "nDCG@10" if metric == "ndcg10" else "Recall@10"
            res = paired_query_bootstrap(
                scores_a,
                scores_b,
                system_a_name=name_a,
                system_b_name=name_b,
                metric_name=m_label,
                n_resamples=10000,
                seed=42,
            )
            key = f"{name_a}_vs_{name_b}_{metric}"
            bootstrap_results[key] = res.to_dict()

            bootstrap_table.append({
                "Comparison": f"{name_a} vs {name_b}",
                "Metric": m_label,
                f"Mean({name_a})": res.mean_a,
                f"Mean({name_b})": res.mean_b,
                "Delta (A - B)": res.mean_diff,
                "95% CI Lower": res.ci_lower_95,
                "95% CI Upper": res.ci_upper_95,
                "p-value": res.p_value,
                "Significant?": "YES" if res.is_significant else "NO",
                "Win %": f"{res.improvement_rate*100:.1f}%",
                "Tie %": f"{res.tie_rate*100:.1f}%",
                "Loss %": f"{res.regression_rate*100:.1f}%",
            })

    df_boot = pd.DataFrame(bootstrap_table)
    print(df_boot.to_string(index=False), flush=True)

    # Save artifacts
    res_dir = "results/fiqa"
    df_summary.to_csv(os.path.join(res_dir, "metrics_summary.csv"), index=False)
    df_boot.to_csv(os.path.join(res_dir, "bootstrap_comparisons.csv"), index=False)
    with open(os.path.join(res_dir, "statistical_validation.json"), "w", encoding="utf-8") as f:
        json.dump(bootstrap_results, f, indent=2)
    with open(os.path.join(res_dir, "adaptive_routing_decisions.jsonl"), "w", encoding="utf-8") as f:
        for r in routing_records:
            f.write(json.dumps(r) + "\n")

    print(f"\nAll FiQA benchmark artifacts successfully saved in {time.perf_counter() - t_start:.1f}s.", flush=True)


if __name__ == "__main__":
    evaluate_fiqa_benchmark()
