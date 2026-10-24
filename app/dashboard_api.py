"""
Dashboard API Data Providers and Utilities.
Provides structured benchmark summaries, run metrics, corpus inventory, and export formats.
"""

import json
import os
from typing import Any
import pandas as pd

SUMMARY_CSV_PATH = "results/summary.csv"
CORPUS_PARQUET_PATH = "data/processed/documents.parquet"
RUNS_DIR = "results/runs"


def get_benchmark_summary_data() -> list[dict[str, Any]]:
    """Load all completed benchmark summary rows as dictionaries."""
    if not os.path.exists(SUMMARY_CSV_PATH):
        return []
    try:
        df = pd.read_csv(SUMMARY_CSV_PATH)
        # Replace NaN with None or defaults for clean JSON serialization
        df = df.fillna({
            "quantization": "none",
            "score_threshold": "none",
            "failure_rate": 0.0,
            "dropped_iterations": 0.0,
        })
        records = df.to_dict(orient="records")
        # Format human-friendly configuration labels
        for r in records:
            label = "Unknown"
            quant = r.get("quantization", "none")
            m = r.get("hnsw_m", 16)
            ef_c = r.get("ef_construct", 128)
            ef = r.get("hnsw_ef", 64)
            cs = r.get("chunk_size", 256)
            ov = r.get("overlap", 32)
            
            if quant == "scalar":
                label = "Baseline D (Scalar INT8)"
            elif quant == "binary":
                label = "Baseline E (Binary 1-bit)"
            elif m == 32 and ef == 128:
                label = "Baseline C (Optimised HNSW m=32)"
            elif ef == 32:
                label = "Optimisation 1 (Fast ef=32)"
            elif cs == 128 and ov == 0:
                label = "Optimisation 2 (Fine 128/0)"
            elif cs == 448:
                label = "Optimisation 3 (Coarse 448/32)"
            elif r.get("exact_match") is not None and r.get("ann_recall_at_10") == 1.0 and r.get("score_threshold") == "none":
                # Check if exact
                if r.get("config_sha") == "60b925f41e28bea3":
                    label = "Baseline A (Exact Search)"
                else:
                    label = "Baseline B (Default HNSW)"
            else:
                label = f"Config {str(r.get('config_sha', ''))[:8]}"
            r["label"] = label
        return records
    except Exception as exc:
        return [{"error": str(exc)}]


def get_corpus_statistics() -> dict[str, Any]:
    """Retrieve verified corpus counts, sizes, and domain breakdown."""
    if not os.path.exists(CORPUS_PARQUET_PATH):
        return {"total_documents": 0, "sources": {}}
    try:
        df = pd.read_parquet(CORPUS_PARQUET_PATH)
        counts = df["source"].value_counts().to_dict()
        size_bytes = os.path.getsize(CORPUS_PARQUET_PATH)
        return {
            "total_documents": len(df),
            "sources": counts,
            "parquet_size_bytes": size_bytes,
            "storage_path": CORPUS_PARQUET_PATH,
        }
    except Exception as exc:
        return {"error": str(exc)}


def get_available_runs() -> list[dict[str, Any]]:
    """List all available run IDs with basic details."""
    if not os.path.exists(RUNS_DIR):
        return []
    runs = []
    for item in os.listdir(RUNS_DIR):
        run_path = os.path.join(RUNS_DIR, item)
        if os.path.isdir(run_path):
            parquet_file = os.path.join(run_path, "queries.parquet")
            query_count = 0
            if os.path.exists(parquet_file):
                try:
                    q_df = pd.read_parquet(parquet_file)
                    query_count = len(q_df)
                except Exception:
                    pass
            runs.append({
                "run_id": item,
                "has_queries_parquet": os.path.exists(parquet_file),
                "query_count": query_count,
            })
    return runs


def get_run_queries(run_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Fetch sample query-level records from a specific run."""
    parquet_file = os.path.join(RUNS_DIR, run_id, "queries.parquet")
    if not os.path.exists(parquet_file):
        return []
    try:
        df = pd.read_parquet(parquet_file)
        if limit:
            df = df.head(limit)
        return df.to_dict(orient="records")
    except Exception as exc:
        return [{"error": str(exc)}]


def get_scifact_benchmark_data() -> list[dict[str, Any]]:
    """Load authentic BEIR SciFact benchmark results."""
    json_path = "results/scifact/results.json"
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        return [{"error": str(exc)}]


def get_qdrant_audit_data() -> dict[str, Any]:
    """Load real Qdrant server audit metrics."""
    audit_path = "results/real_qdrant/audit_metrics.json"
    if not os.path.exists(audit_path):
        return {}
    try:
        with open(audit_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        return {"error": str(exc)}


def get_load_test_data() -> dict[str, Any]:
    """Load systems concurrency ladder sweep results."""
    load_path = "results/load_tests/concurrency_sweep.json"
    if not os.path.exists(load_path):
        load_path = "results/load_test_results.json"
    if not os.path.exists(load_path):
        return {}
    try:
        with open(load_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        return {"error": str(exc)}


# =====================================================================
# Phase 5 Research Dashboard Endpoints
# =====================================================================

def get_phase5_datasets_data() -> dict[str, Any]:
    """Returns cross-domain benchmark dataset metrics (SciFact, NFCorpus, FiQA)."""
    datasets = {}
    
    # SciFact
    scifact_csv = "results/scifact/corrected_results.csv"
    if os.path.exists(scifact_csv):
        df_sci = pd.read_csv(scifact_csv)
        datasets["scifact"] = {
            "name": "BEIR/SciFact",
            "domain": "Biomedical Scientific Claims",
            "corpus_docs": 5183,
            "test_queries": 300,
            "metrics": df_sci.fillna("").to_dict(orient="records"),
        }

    # NFCorpus
    nf_summary = "results/nfcorpus/metrics_summary.csv"
    if os.path.exists(nf_summary):
        df_nf = pd.read_csv(nf_summary)
        datasets["nfcorpus"] = {
            "name": "BEIR/NFCorpus",
            "domain": "Nutrition & Clinical Medicine",
            "corpus_docs": 3633,
            "test_queries": 323,
            "metrics": df_nf.fillna("").to_dict(orient="records"),
        }

    # FiQA
    fiqa_summary = "results/fiqa/metrics_summary.csv"
    fiqa_json = "results/fiqa/bm25_results.json"
    if os.path.exists(fiqa_summary):
        df_fiqa = pd.read_csv(fiqa_summary)
        datasets["fiqa"] = {
            "name": "BEIR/FiQA",
            "domain": "Financial Opinion QA",
            "corpus_docs": 57638,
            "test_queries": 648,
            "metrics": df_fiqa.fillna("").to_dict(orient="records"),
        }
    elif os.path.exists(fiqa_json):
        with open(fiqa_json, "r", encoding="utf-8") as f:
            fiqa_data = json.load(f)
        datasets["fiqa"] = {
            "name": "BEIR/FiQA",
            "domain": "Financial Opinion QA",
            "corpus_docs": 57638,
            "test_queries": 648,
            "metrics": [fiqa_data],
        }

    return datasets


def get_phase5_bootstrap_data(dataset: str = "scifact") -> list[dict[str, Any]]:
    """Loads paired query-level bootstrap comparison tables with 95% CIs."""
    target_path = f"results/{dataset}/bootstrap_comparisons.csv"
    if not os.path.exists(target_path):
        target_path = "results/scifact/bootstrap_comparisons.csv"
    if not os.path.exists(target_path):
        return []
    try:
        df = pd.read_csv(target_path)
        return df.fillna("").to_dict(orient="records")
    except Exception as exc:
        return [{"error": str(exc)}]


def get_phase5_routing_data(dataset: str = "scifact", limit: int = 100) -> list[dict[str, Any]]:
    """Loads per-query adaptive routing decisions and signals."""
    jsonl_path = f"results/{dataset}/adaptive_routing_decisions.jsonl"
    if not os.path.exists(jsonl_path):
        jsonl_path = "results/scifact/adaptive_routing_decisions.jsonl"
    if not os.path.exists(jsonl_path):
        return []
    records = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                records.append(json.loads(line))
        return records
    except Exception as exc:
        return [{"error": str(exc)}]


def get_phase5_failure_cases() -> list[dict[str, Any]]:
    """Loads forensic failure case analysis where Hybrid or Adaptive degrades."""
    fail_path = "results/scifact/failure_cases.json"
    if not os.path.exists(fail_path):
        return []
    try:
        with open(fail_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        return [{"error": str(exc)}]


def get_phase5_ablations() -> dict[str, Any]:
    """Loads all Stage 5 ablation study datasets."""
    abl_dir = "results/ablations"
    results = {}
    for f in ["fusion_parameter_ablation.csv", "topk_cutoff_ablation.csv", "adaptive_feature_ablation.csv", "quantization_rescore_ablation.csv"]:
        fp = os.path.join(abl_dir, f)
        key = f.replace(".csv", "")
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            results[key] = df.fillna("").to_dict(orient="records")
    return results


def get_phase5_service_perf() -> dict[str, Any]:
    """Loads Stage 6 real service HTTP benchmark results."""
    json_path = "results/service_performance/http_benchmark_results.json"
    if not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        return {"error": str(exc)}


def get_canonical_registry_data() -> dict[str, Any]:
    """Loads Phase 6 Canonical Results Registry."""
    json_path = "results/canonical_results_registry.json"
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            return {"error": str(exc)}
    return {}


def get_adaptive_runtime_audit_data() -> dict[str, Any]:
    """Loads Phase 6 Task 3 Runtime Cost & Timing Breakdown Audit."""
    json_path = "results/service_performance/adaptive_runtime_audit.json"
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            return {"error": str(exc)}
    return {}


def get_generation_audit_data() -> dict[str, Any]:
    """Loads Phase 6 Task 6 RAG Generation Audit & Reality Disclosure."""
    json_path = "results/qa/generation_verification_audit.json"
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            return {"error": str(exc)}
    return {}


def get_scientific_integrity_audit_data() -> dict[str, Any]:
    """Loads Final Milestone Priority 1 Scientific Integrity Audit data."""
    json_path = "results/scientific_integrity_audit.json"
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            return {"error": str(exc)}
    return {}


def get_generation_evaluation_data() -> dict[str, Any]:
    """Loads Priority 4 Generative Answer Evaluation Data."""
    json_path = "results/qa/generation_evaluation_results.json"
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            return {"error": str(exc)}
    return {}





