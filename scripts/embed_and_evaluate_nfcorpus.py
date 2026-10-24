"""
NFCorpus Embedding, Adaptive Retrieval, and Statistical Validation Pipeline.
Operates strictly with capped thread usage (torch.set_num_threads(2)) and caches
all embeddings to disk immediately as .npy arrays.
Evaluates BM25, Dense Exact, Fixed Hybrid RRF, and Adaptive Router across official BEIR NFCorpus.
"""

from collections import defaultdict
import json
import os
import time
import numpy as np
import pandas as pd
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

# Cap PyTorch CPU threads to protect user hardware from thermal throttling
torch.set_num_threads(2)
torch.set_num_interop_threads(1)


def embed_and_cache_nfcorpus():
    nf_dir = "data/beir/nfcorpus"
    results_dir = "results/nfcorpus"
    os.makedirs(results_dir, exist_ok=True)

    corpus_emb_path = os.path.join(results_dir, "corpus_embeddings.npy")
    query_emb_path = os.path.join(results_dir, "query_embeddings.npy")
    corpus_ids_path = os.path.join(results_dir, "corpus_ids.json")
    query_ids_path = os.path.join(results_dir, "query_ids.json")

    # Load corpus
    corpus_ids = []
    corpus_texts = []
    corpus_dict = {}
    with open(os.path.join(nf_dir, "corpus.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            cid = item["_id"]
            text = (item.get("title", "") + " " + item.get("text", "")).strip()
            corpus_ids.append(cid)
            corpus_texts.append(text)
            corpus_dict[cid] = text

    # Load test queries and test qrels
    test_qrels = defaultdict(set)
    with open(os.path.join(nf_dir, "qrels/test.tsv"), "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3 and int(parts[2]) > 0:
                test_qrels[parts[0]].add(parts[1])

    queries_dict = {}
    with open(os.path.join(nf_dir, "queries.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            queries_dict[item["_id"]] = item["text"]

    test_qids = [qid for qid in test_qrels if qid in queries_dict]
    test_qtexts = [queries_dict[qid] for qid in test_qids]

    # Load dev qrels for calibration
    dev_qrels = defaultdict(set)
    with open(os.path.join(nf_dir, "qrels/dev.tsv"), "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3 and int(parts[2]) > 0:
                dev_qrels[parts[0]].add(parts[1])

    if not (os.path.exists(corpus_emb_path) and os.path.exists(query_emb_path)):
        print(f"Generating dense embeddings for NFCorpus (Docs={len(corpus_ids)}, Queries={len(test_qids)})...", flush=True)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        t0 = time.perf_counter()
        corpus_vecs = model.encode(
            corpus_texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        print(f"Corpus encoded in {time.perf_counter() - t0:.2f}s. Shape: {corpus_vecs.shape}", flush=True)
        
        query_vecs = model.encode(
            test_qtexts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        print(f"Test queries encoded in {time.perf_counter() - t0:.2f}s. Shape: {query_vecs.shape}", flush=True)

        np.save(corpus_emb_path, corpus_vecs)
        np.save(query_emb_path, query_vecs)
        with open(corpus_ids_path, "w", encoding="utf-8") as f:
            json.dump(corpus_ids, f)
        with open(query_ids_path, "w", encoding="utf-8") as f:
            json.dump(test_qids, f)
        print(f"Cached NFCorpus embeddings to disk ({corpus_emb_path})", flush=True)
    else:
        print("Using existing cached NFCorpus dense embeddings.", flush=True)
        corpus_vecs = np.load(corpus_emb_path)
        query_vecs = np.load(query_emb_path)

    return {
        "corpus_ids": corpus_ids,
        "corpus_vecs": corpus_vecs,
        "corpus_dict": corpus_dict,
        "test_qids": test_qids,
        "test_qvecs": query_vecs,
        "queries_dict": queries_dict,
        "test_qrels": test_qrels,
        "dev_qrels": dev_qrels,
    }


def calibrate_nfcorpus_dev(bm25: BM25Okapi, queries_dict: dict, dev_qrels: dict) -> AdaptiveThresholds:
    print("\n--- Calibrating Adaptive Router on NFCorpus Dev Set ---", flush=True)
    dev_qids = [qid for qid in dev_qrels if qid in queries_dict]
    print(f"Calibrating on {len(dev_qids)} independent development queries...", flush=True)

    records = []
    for qid in dev_qids:
        qtext = queries_dict[qid]
        gold_set = dev_qrels[qid]
        tokens = tokenize_text(qtext)
        idfs = [bm25.idf.get(t, 0.0) for t in tokens]
        max_idf = max(idfs) if idfs else 0.0

        hits = bm25.search(qtext, top_k=10)
        if not hits:
            continue
        top_doc, top_score = hits[0]
        runner_score = hits[1][1] if len(hits) > 1 else 0.0
        margin = (top_score - runner_score) / (top_score + 1e-6)
        is_hit1 = top_doc in gold_set
        records.append({
            "qid": qid,
            "max_idf": max_idf,
            "margin": margin,
            "top_score": top_score,
            "is_hit1": is_hit1,
        })

    df = pd.DataFrame(records)
    best_tau_margin = 0.25
    best_tau_idf = 5.0
    best_coverage = 0.0

    for m_cand in [0.15, 0.20, 0.25, 0.30]:
        for idf_cand in [4.0, 5.0, 6.0]:
            subset = df[(df["margin"] >= m_cand) & (df["max_idf"] >= idf_cand) & (df["top_score"] >= 6.0)]
            if len(subset) > 15:
                p1 = subset["is_hit1"].mean()
                coverage = len(subset) / len(df)
                if p1 >= 0.50 and coverage > best_coverage:
                    best_coverage = coverage
                    best_tau_margin = m_cand
                    best_tau_idf = idf_cand

    print(f"NFCorpus Calibration: tau_margin={best_tau_margin}, tau_min_idf={best_tau_idf} (Dev Coverage={best_coverage*100:.1f}%)", flush=True)
    return AdaptiveThresholds(
        tau_lexical_margin=best_tau_margin,
        tau_min_idf=best_tau_idf,
        tau_dense_confident=0.70,
        tau_bm25_weak=6.0,
        k_rrf=60,
    )


def main():
    print("=================================================================", flush=True)
    print("  RAGScaleBench Phase 5: NFCorpus Cross-Domain Benchmark         ", flush=True)
    print("=================================================================", flush=True)

    data = embed_and_cache_nfcorpus()
    corpus_ids = data["corpus_ids"]
    corpus_vecs = data["corpus_vecs"]
    corpus_dict = data["corpus_dict"]
    test_qids = data["test_qids"]
    test_qvecs = data["test_qvecs"]
    queries_dict = data["queries_dict"]
    test_qrels = data["test_qrels"]
    dev_qrels = data["dev_qrels"]

    # 1. Fit BM25 Okapi index
    print(f"\nBuilding BM25 Okapi index for {len(corpus_ids)} documents...", flush=True)
    bm25 = BM25Okapi()
    bm25.fit([(cid, corpus_dict[cid]) for cid in corpus_ids])

    # 2. Dense search function
    corpus_vecs_t = corpus_vecs.T
    def dense_search(query_vec: np.ndarray, top_k: int = 50) -> list[tuple[str, float]]:
        sims = np.dot(query_vec, corpus_vecs_t)
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(corpus_ids[idx], float(sims[idx])) for idx in top_idx]

    # 3. Calibrate on dev set
    thresholds = calibrate_nfcorpus_dev(bm25, queries_dict, dev_qrels)

    # 4. Initialize Policies
    policies = {
        "BM25": BM25Policy(bm25),
        "Dense_Exact": DensePolicy(dense_search),
        "Fixed_Hybrid": FixedHybridPolicy(bm25, dense_search, k_rrf=60),
        "Adaptive_Router": AdaptiveRouterPolicy(bm25, dense_search, thresholds),
    }

    # 5. Evaluate on locked test set
    print(f"\n--- Evaluating Policies on {len(test_qids)} Locked Test Queries ---", flush=True)
    per_query_metrics = {name: {"recall10": {}, "ndcg10": {}, "mrr": {}} for name in policies}
    routing_records = []
    dense_searches_executed = 0

    for idx, qid in enumerate(test_qids):
        qtext = queries_dict[qid]
        qvec = test_qvecs[idx]
        gold_set = test_qrels[qid]

        for name, policy in policies.items():
            res = policy.retrieve(qtext, query_vec=qvec, top_k=50)
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
    summary_rows = []
    for name in policies:
        r10_mean = float(np.mean(list(per_query_metrics[name]["recall10"].values())))
        ndcg10_mean = float(np.mean(list(per_query_metrics[name]["ndcg10"].values())))
        mrr_mean = float(np.mean(list(per_query_metrics[name]["mrr"].values())))
        summary_rows.append({
            "Policy": name,
            "Recall@10": round(r10_mean, 4),
            "nDCG@10": round(ndcg10_mean, 4),
            "MRR": round(mrr_mean, 4),
        })

    df_summary = pd.DataFrame(summary_rows)
    print("\nNFCorpus Evaluation Metrics:")
    print(df_summary.to_string(index=False), flush=True)

    dense_saved = len(test_qids) - dense_searches_executed
    dense_saved_pct = (dense_saved / len(test_qids)) * 100.0
    print(f"\nAdaptive Routing on NFCorpus: Dense Saved = {dense_saved}/{len(test_qids)} ({dense_saved_pct:.1f}%)", flush=True)

    # 6. Paired Bootstrap Statistical Validation (B=10,000 resamples, seed=42)
    print("\n--- Running Paired Query Bootstrap Validation (B=10,000) ---", flush=True)
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
                "Delta": res.mean_diff,
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
    results_dir = "results/nfcorpus"
    with open(os.path.join(results_dir, "statistical_validation.json"), "w", encoding="utf-8") as f:
        json.dump(bootstrap_results, f, indent=2)

    with open(os.path.join(results_dir, "adaptive_routing_decisions.jsonl"), "w", encoding="utf-8") as f:
        for rec in routing_records:
            f.write(json.dumps(rec) + "\n")

    df_boot.to_csv(os.path.join(results_dir, "bootstrap_comparisons.csv"), index=False)
    df_summary.to_csv(os.path.join(results_dir, "metrics_summary.csv"), index=False)
    print(f"\nSaved NFCorpus benchmark artifacts to {results_dir}/", flush=True)


if __name__ == "__main__":
    main()
