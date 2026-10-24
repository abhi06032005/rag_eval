"""Replay fusion/router ablations on new ranked outputs; no replay timings are benchmarked."""

import argparse
import ast
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ragbench.evaluation.bm25 import BM25Okapi, tokenize_text
from ragbench.retrieval.policies import AdaptiveRouterPolicy, AdaptiveThresholds, rrf_fuse
from scripts.full_beir_reevaluation import OUT, dump, load_data, metrics, statistics


def original_calibration(name, bm, queries, rels):
    paths = {
        "scifact": ("scripts/run_stage2_and_stage4_scifact.py", "calibrate_thresholds_on_dev"),
        "nfcorpus": ("scripts/embed_and_evaluate_nfcorpus.py", "calibrate_nfcorpus_dev"),
        "fiqa": ("scripts/embed_and_evaluate_fiqa.py", "calibrate_fiqa_dev"),
    }
    path, fn = paths[name]
    # Compile only the audited calibration function, avoiding module-level runtime mutations.
    module = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == fn)
    ns = {
        "BM25Okapi": BM25Okapi,
        "AdaptiveThresholds": AdaptiveThresholds,
        "tokenize_text": tokenize_text,
        "np": np,
        "pd": pd,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
    return ns[fn](bm, queries, {q: {d for d, g in ds.items() if g > 0} for q, ds in rels.items()})


def main(name):
    root = OUT / name
    corpus, queries, splits = load_data(name)
    gold = splits["test"]
    with (root / "bm25_index.pkl").open("rb") as f:
        bm = pickle.load(f)  # only this run's locally generated file
    cal_split = "train" if name == "scifact" else "dev"
    historical = original_calibration(name, bm, queries, splits[cal_split])
    test_texts = {" ".join(queries[q].lower().split()) for q in gold}
    clean = {q: r for q, r in splits[cal_split].items() if " ".join(queries[q].lower().split()) not in test_texts}
    clean_cfg = original_calibration(name, bm, queries, clean)
    dump(
        root / "calibration.json",
        {
            "source_split": cal_split,
            "historical_thresholds": vars(historical),
            "deduplicated_thresholds": vars(clean_cfg),
            "queries_in_split": len(splits[cal_split]),
            "removed_query_ids": sorted(set(splits[cal_split]) - set(clean)),
            "fiqa_legacy_limit": 300 if name == "fiqa" else None,
            "qualification": "Original calibration selection replay; descriptive, not a preregistered confirmatory analysis. FiQA code uses first 300 dev qrels queries, despite earlier report claiming 500.",
        },
    )
    branches = {}
    with gzip.open(root / "per_query.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["depth"] == 250 and r["system"] in ("BM25", "Dense_Exact"):
                branches.setdefault(r["query_id"], {})[r["system"]] = r["rankings"]
    records = []
    for q in sorted(gold):
        b = branches[q]["BM25"]
        d = branches[q]["Dense_Exact"]

        # Replay the actual router implementation with cached searches; thresholds/features unchanged.
        class ReplayBM:
            idf = bm.idf

            def search(self, text, top_k):
                return b[:top_k]

        for depth in (50, 250):
            configurations = {
                f"RRF_k{k}": (rrf_fuse(b[:depth], d[:depth], k_rrf=k, top_k=depth), "hybrid") for k in (10, 30, 60, 100)
            }
            for label, cfg in [
                ("Router_historical", historical),
                ("Router_deduplicated_calibration", clean_cfg),
                ("Router_fixed_defaults", AdaptiveThresholds()),
                ("Router_no_margin", AdaptiveThresholds(tau_lexical_margin=0.0)),
                ("Router_no_idf", AdaptiveThresholds(tau_min_idf=0.0)),
                ("Router_always_hybrid", AdaptiveThresholds(tau_lexical_margin=2.0, tau_dense_confident=2.0)),
            ]:
                router = AdaptiveRouterPolicy(ReplayBM(), lambda v, k: d[:k], thresholds=cfg, candidate_depth=depth)
                res = router.retrieve(queries[q], [0.0], top_k=depth)
                configurations[label] = (res.retrieved_docs, res.route_taken)
            for label, (hits, route) in configurations.items():
                score = metrics([x[0] for x in hits], gold[q])
                score.pop("MRR@250")
                records.append(
                    {
                        "query_id": q,
                        "system": label,
                        "depth": depth,
                        "route": route,
                        "rankings": hits[:10],
                        "metrics": score,
                        "evidence": "REPLAY_OF_NEW_RANKINGS; no latency inference",
                    }
                )
    with gzip.open(root / "ablations_per_query.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    summary = []
    for depth in (50, 250):
        for system in sorted({r["system"] for r in records}):
            rows = [r for r in records if r["system"] == system and r["depth"] == depth]
            summary.append(
                {
                    "depth": depth,
                    "system": system,
                    "metrics": {k: float(np.mean([r["metrics"][k] for r in rows])) for k in rows[0]["metrics"]},
                    "lexical_routes": sum(r["route"] == "bm25" for r in rows),
                    "n": len(rows),
                }
            )
    dump(root / "ablations.json", summary)
    matched = [
        r
        for r in records
        if r["depth"] == 250
        and r["system"] in ("RRF_k60", "Router_historical", "Router_deduplicated_calibration", "Router_fixed_defaults")
    ]
    dump(root / "calibrated_statistics.json", statistics(matched))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    main(p.parse_args().dataset)
