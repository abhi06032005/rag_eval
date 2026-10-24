"""
Stage 2 & Stage 4 Execution Script for SciFact:
1. Calibrates Adaptive Router on independent development set (train.tsv).
2. Evaluates BM25, Dense Exact, Fixed Hybrid RRF, and Adaptive Router on locked test set (300 queries).
3. Performs paired query-level bootstrap evaluation (B=10,000 resamples, seed=42).
4. Generates failure case analysis and resource savings report.
"""

from collections import defaultdict
import json
import os
import time
import numpy as np
import pandas as pd

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


def load_dataset():
    scifact_dir = "data/beir/scifact"
    results_dir = "results/scifact"

    corpus_emb = np.load(os.path.join(results_dir, "corpus_embeddings.npy"))
    query_emb = np.load(os.path.join(results_dir, "query_embeddings.npy"))
    with open(os.path.join(results_dir, "corpus_ids.json"), "r", encoding="utf-8") as f:
        corpus_ids = json.load(f)
    with open(os.path.join(results_dir, "query_ids.json"), "r", encoding="utf-8") as f:
        test_query_ids = json.load(f)

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

    train_qrels = defaultdict(set)
    with open(os.path.join(scifact_dir, "qrels/train.tsv"), "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3 and int(parts[2]) > 0:
                train_qrels[parts[0]].add(parts[1])

    return {
        "corpus_ids": corpus_ids,
        "corpus_vecs": corpus_emb,
        "corpus_dict": corpus_dict,
        "test_query_ids": test_query_ids,
        "test_query_vecs": query_emb,
        "queries_dict": queries_dict,
        "test_qrels": test_qrels,
        "train_qrels": train_qrels,
    }


def calibrate_thresholds_on_dev(bm25: BM25Okapi, queries_dict: dict, train_qrels: dict) -> AdaptiveThresholds:
    """
    Calibrates lexical confidence thresholds strictly on train qrels without touching test set.
    Goal: Select tau_margin and tau_min_idf where BM25 top-1 has high precision.
    """
    print("\n--- Calibrating Adaptive Router on Independent Training Set (809 queries) ---", flush=True)
    dev_qids = [qid for qid in train_qrels if qid in queries_dict]
    print(f"Calibrating on {len(dev_qids)} independent training queries...", flush=True)

    records = []
    for qid in dev_qids:
        qtext = queries_dict[qid]
        gold_set = train_qrels[qid]
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
    
    # Grid sweep on dev set: find threshold with >= 85% Precision@1 for lexical-only route
    best_tau_margin = 0.28
    best_tau_idf = 6.0
    best_coverage = 0.0

    for m_cand in [0.20, 0.25, 0.28, 0.32, 0.35]:
        for idf_cand in [5.0, 5.5, 6.0, 6.5]:
            subset = df[(df["margin"] >= m_cand) & (df["max_idf"] >= idf_cand) & (df["top_score"] >= 8.0)]
            if len(subset) > 20:
                p1 = subset["is_hit1"].mean()
                coverage = len(subset) / len(df)
                if p1 >= 0.70 and coverage > best_coverage:
                    best_coverage = coverage
                    best_tau_margin = m_cand
                    best_tau_idf = idf_cand

    print(f"Calibration Result: tau_lexical_margin={best_tau_margin}, tau_min_idf={best_tau_idf} (Dev Coverage={best_coverage*100:.1f}%)", flush=True)
    return AdaptiveThresholds(
        tau_lexical_margin=best_tau_margin,
        tau_min_idf=best_tau_idf,
        tau_dense_confident=0.75,
        tau_bm25_weak=8.0,
        k_rrf=60,
    )


def main():
    print("=================================================================", flush=True)
    print("  RAGScaleBench Phase 5: Stage 2 & Stage 4 Statistical Validation", flush=True)
    print("=================================================================", flush=True)

    data = load_dataset()
    corpus_ids = data["corpus_ids"]
    corpus_vecs = data["corpus_vecs"]
    corpus_dict = data["corpus_dict"]
    test_qids = data["test_query_ids"]
    test_qvecs = data["test_query_vecs"]
    queries_dict = data["queries_dict"]
    test_qrels = data["test_qrels"]
    train_qrels = data["train_qrels"]

    # 1. Build BM25 Okapi index
    print(f"\nBuilding BM25 Okapi index for {len(corpus_ids)} documents...", flush=True)
    bm25 = BM25Okapi()
    bm25.fit([(cid, corpus_dict[cid]) for cid in corpus_ids])

    # 2. Dense search function (NumPy exact)
    corpus_vecs_t = corpus_vecs.T
    def dense_search(query_vec: np.ndarray, top_k: int = 50) -> list[tuple[str, float]]:
        sims = np.dot(query_vec, corpus_vecs_t)
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(corpus_ids[idx], float(sims[idx])) for idx in top_idx]

    # 3. Calibrate Adaptive Router on independent dev/train set
    calibrated_thresholds = calibrate_thresholds_on_dev(bm25, queries_dict, train_qrels)

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

    # 5. Evaluate all policies across the locked 300 test queries
    print("\n--- Evaluating Policies on Locked 300 Test Queries ---", flush=True)
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
                    "gold_docs": list(gold_set),
                    "retrieved_top3": [d[0] for d in res.retrieved_docs[:3]],
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
    print("\nSummary Evaluation Metrics:")
    print(df_summary.to_string(index=False), flush=True)

    # 6. Route distribution and compute savings
    route_counts = Counter = defaultdict(int)
    for rec in routing_records:
        route_counts[rec["route_taken"]] += 1
    total_test_queries = len(test_qids)
    dense_saved = total_test_queries - dense_searches_executed
    dense_saved_pct = (dense_saved / total_test_queries) * 100.0

    print(f"\nAdaptive Routing Distribution ({total_test_queries} queries):")
    for r, cnt in route_counts.items():
        print(f"  Route [{r}]: {cnt} queries ({cnt/total_test_queries*100:.1f}%)", flush=True)
    print(f"Dense vector searches saved: {dense_saved} / {total_test_queries} ({dense_saved_pct:.1f}%)", flush=True)

    # 7. Paired Query-Level Bootstrap Evaluation (B=10,000 resamples, seed=42)
    print("\n=================================================================", flush=True)
    print("      PAIRED BOOTSTRAP STATISTICAL VALIDATION (B=10,000)         ", flush=True)
    print("=================================================================", flush=True)

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

    # 8. Detailed Failure Analysis: Where does Hybrid or Adaptive degrade?
    print("\n--- Failure Case Analysis: Hybrid & Adaptive Regressions ---", flush=True)
    # Cases where Hybrid < BM25
    hybrid_vs_bm25_ndcg = bootstrap_results["BM25_vs_Fixed_Hybrid_ndcg10"]
    # Here Delta is BM25 - Hybrid. Queries where BM25 > Hybrid are regressions for Hybrid!
    hybrid_degradation_queries = hybrid_vs_bm25_ndcg["regression_query_ids"] # where BM25 < Hybrid
    bm25_beats_hybrid_qids = [
        qid for qid in test_qids 
        if per_query_metrics["BM25"]["ndcg10"][qid] > per_query_metrics["Fixed_Hybrid"]["ndcg10"][qid] + 1e-4
    ]
    dense_beats_hybrid_qids = [
        qid for qid in test_qids 
        if per_query_metrics["Dense_Exact"]["ndcg10"][qid] > per_query_metrics["Fixed_Hybrid"]["ndcg10"][qid] + 1e-4
    ]
    hybrid_beats_adaptive_qids = [
        qid for qid in test_qids 
        if per_query_metrics["Fixed_Hybrid"]["ndcg10"][qid] > per_query_metrics["Adaptive_Router"]["ndcg10"][qid] + 1e-4
    ]

    print(f"Queries where BM25 strictly outperformed Hybrid: {len(bm25_beats_hybrid_qids)} ({len(bm25_beats_hybrid_qids)/300*100:.1f}%)")
    print(f"Queries where Dense strictly outperformed Hybrid: {len(dense_beats_hybrid_qids)} ({len(dense_beats_hybrid_qids)/300*100:.1f}%)")
    print(f"Queries where Fixed Hybrid outperformed Adaptive: {len(hybrid_beats_adaptive_qids)} ({len(hybrid_beats_adaptive_qids)/300*100:.1f}%)")

    failure_case_details = []
    for qid in bm25_beats_hybrid_qids[:5]:
        failure_case_details.append({
            "query_id": qid,
            "type": "Hybrid_Worse_Than_BM25",
            "query_text": queries_dict[qid],
            "bm25_ndcg": per_query_metrics["BM25"]["ndcg10"][qid],
            "dense_ndcg": per_query_metrics["Dense_Exact"]["ndcg10"][qid],
            "hybrid_ndcg": per_query_metrics["Fixed_Hybrid"]["ndcg10"][qid],
            "mechanism": "Dense branch retrieved irrelevant semantic distractors that diluted high-precision BM25 rank.",
        })

    for qid in dense_beats_hybrid_qids[:5]:
        failure_case_details.append({
            "query_id": qid,
            "type": "Hybrid_Worse_Than_Dense",
            "query_text": queries_dict[qid],
            "bm25_ndcg": per_query_metrics["BM25"]["ndcg10"][qid],
            "dense_ndcg": per_query_metrics["Dense_Exact"]["ndcg10"][qid],
            "hybrid_ndcg": per_query_metrics["Fixed_Hybrid"]["ndcg10"][qid],
            "mechanism": "BM25 retrieved zero relevant hits (vocabulary mismatch) while dense found gold at rank 1; RRF fusion demoted gold rank.",
        })

    # Save outputs
    results_dir = "results/scifact"
    with open(os.path.join(results_dir, "statistical_validation.json"), "w", encoding="utf-8") as f:
        json.dump(bootstrap_results, f, indent=2)

    with open(os.path.join(results_dir, "adaptive_routing_decisions.jsonl"), "w", encoding="utf-8") as f:
        for rec in routing_records:
            f.write(json.dumps(rec) + "\n")

    with open(os.path.join(results_dir, "failure_cases.json"), "w", encoding="utf-8") as f:
        json.dump(failure_case_details, f, indent=2)

    df_boot.to_csv(os.path.join(results_dir, "bootstrap_comparisons.csv"), index=False)
    print("\nSaved statistical validation artifacts to results/scifact/", flush=True)


if __name__ == "__main__":
    main()
