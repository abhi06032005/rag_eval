"""Build the registry only from evidence, with explicit depth and cutoff provenance."""

import csv
import gzip
import json
from pathlib import Path

import numpy as np

from scripts.full_beir_reevaluation import OUT, dump, load_data, metrics, sha


def reconcile_and_register():
    records = []
    for name in ("scifact", "nfcorpus", "fiqa"):
        root = OUT / name
        summaries = json.loads((root / "metrics.json").read_text())
        corpus, _, splits = load_data(name)
        qrels = splits["test"]
        corpus_ids = {row["_id"] for row in corpus}
        groups = {}
        with gzip.open(root / "per_query.jsonl.gz", "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                key = (r["depth"], r["system"])
                assert r["query_id"] not in groups.setdefault(key, {})
                assert all(doc_id in corpus_ids for doc_id, _ in r["rankings"])
                score = metrics([d for d, s in r["rankings"]], qrels[r["query_id"]])
                groups[key][r["query_id"]] = score
        for s in summaries:
            depth = s["candidate_depth_per_branch"]
            rows = groups[depth, s["system"]]
            assert set(rows) == set(qrels), "incomplete query population"
            recomputed = {
                k: float(np.mean([r[k] for r in rows.values()]))
                for k in (
                    "Recall@1",
                    "Recall@5",
                    "Recall@10",
                    "Precision@1",
                    "Precision@5",
                    "Precision@10",
                    "nDCG@10",
                    "MRR@10",
                )
            }
            for k, v in recomputed.items():
                assert abs(v - s["metrics"][k]) < 1e-12
            records.append(
                {
                    "experiment_id": f"REEXEC-{name.upper()}-{s['system']}-CAND{depth}",
                    "dataset_name": "BEIR/" + name,
                    "split": "test",
                    "num_queries": len(rows),
                    "num_docs": len(corpus),
                    "backend": s["system"],
                    "model_name": s["model"],
                    "retrieval_depth_k_cand": depth,
                    "retrieval_top_k": depth,
                    "evaluation_cutoff": 10,
                    **recomputed,
                    "status": "NEW_EXECUTION",
                    "rankings_sha256": sha(root / "per_query.jsonl.gz"),
                    "rankings_path": str(root / "per_query.jsonl.gz"),
                    "latency_p50_ms": s["p50_ms"],
                    "latency_p95_ms": s["p95_ms"],
                    "latency_p99_ms": s["p99_ms"],
                    "latency_scope": s["latency_scope"],
                }
            )
    # Preserve the legacy experiment ID, but derive every metric from its correct 250-candidate rankings.
    source = OUT / "scifact_reconciliation/per_query.jsonl.gz"
    qrels = load_data("scifact")[2]["test"]
    selected = {}
    with gzip.open(source, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["configuration"] == "reconstructed_branch_D250":
                selected[r["query_id"]] = metrics([d for d, s in r["rankings"]], qrels[r["query_id"]])
    assert set(selected) == set(qrels)
    mean = {k: float(np.mean([s[k] for s in selected.values()])) for k in next(iter(selected.values()))}
    mean["MRR@50"] = mean.pop("MRR@250")
    records.append(
        {
            "experiment_id": "EXP-SCIFACT-HYBRID-D50",
            "dataset_name": "BEIR/SciFact",
            "split": "test",
            "num_queries": 300,
            "backend": "Fixed_Hybrid",
            "model_name": "BM25 + all-MiniLM-L6-v2",
            "retrieval_depth_k_cand": 250,
            "retrieval_top_k": 50,
            "evaluation_cutoff": 10,
            **mean,
            "status": "RECONSTRUCTED_WITH_ARCHIVED_EMBEDDINGS",
            "rankings_path": str(source),
            "rankings_sha256": sha(source),
            "reconciliation_notes": "Legacy D50 means returned cutoff 50, NOT candidate depth 50. Source branches rebuilt to 250; no latency inferred from old summary.",
        }
    )
    dump(
        "results/canonical_results_registry.json",
        {
            "registry_version": "2.0-evidence-derived",
            "checksums": json.loads((OUT / "dataset_verification.json").read_text()),
            "non_inferiority_analysis": {},
            "multiple_comparison_corrections": {},
            "paired_statistics": {
                name: f"results/recomputed/{name}/statistics.json" for name in ("scifact", "nfcorpus", "fiqa")
            },
            "records": records,
            "retired_registry": "results/recomputed/archived_canonical_results_registry.json",
            "method": "All quality values recomputed from ranked documents and official qrels; no hard-coded results.",
        },
    )
    fields = list(dict.fromkeys(k for r in records for k in r))
    with Path("results/canonical_results_registry.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    reconcile_and_register()
