"""Reconstruct SciFact candidate-depth regimes; never infer a missing tail from top-50 output."""

import gzip
import json
import time
from pathlib import Path

import numpy as np

from ragbench.evaluation.bm25 import BM25Okapi
from ragbench.retrieval.policies import rrf_fuse
from scripts.full_beir_reevaluation import OUT, dump, load_data, metrics, sha


def main():
    root = OUT / "scifact_reconciliation"
    root.mkdir(exist_ok=True, parents=True)
    corpus, queries, splits = load_data("scifact")
    gold = splits["test"]
    source = Path("results/scifact/per_query_rankings.jsonl")
    archived = {r["query_id"]: r["rankings"] for r in map(json.loads, source.read_text().splitlines())}
    assert set(archived) == set(gold)
    bm = BM25Okapi()
    cids = [c["_id"] for c in corpus]
    bm.fit([(c["_id"], (c.get("title", "") + " " + c["text"]).strip()) for c in corpus])
    old = Path("results/scifact")
    vec = np.load(old / "corpus_embeddings.npy")
    qvec = np.load(old / "query_embeddings.npy")
    assert json.loads((old / "corpus_ids.json").read_text()) == cids
    qids = json.loads((old / "query_ids.json").read_text())
    assert set(qids) == set(gold)
    per = []
    groups = {}
    mismatch = []
    for q, v in zip(qids, qvec):
        b = bm.search(queries[q], 250)
        scores = vec @ v
        order = np.argsort(scores)[::-1][:250]
        d = [(cids[i], float(scores[i])) for i in order]
        if [x[0] for x in b[:50]] != [x[0] for x in archived[q]["BM25"]] or [x[0] for x in d[:50]] != [
            x[0] for x in archived[q]["Dense_Exact"]
        ]:
            mismatch.append(q)
        configurations = {
            "stored_Hybrid_Exact_RRF_D100": archived[q]["Hybrid_Exact_RRF"],
            "refused_stored_branches_D50": rrf_fuse(archived[q]["BM25"], archived[q]["Dense_Exact"], top_k=50),
        }
        for depth in (50, 100, 250):
            configurations[f"reconstructed_branch_D{depth}"] = rrf_fuse(b[:depth], d[:depth], top_k=50)
        for label, hits in configurations.items():
            score = metrics([x[0] for x in hits], gold[q])
            score["MRR@50"] = score.pop("MRR@250")
            groups.setdefault(label, []).append(score)
            per.append(
                {
                    "query_id": q,
                    "configuration": label,
                    "rankings": hits,
                    "metrics": score,
                    "evidence": "RECOMPUTED_ARCHIVED_RANKINGS"
                    if "stored" in label
                    else "NEW_SEARCH_WITH_ARCHIVED_EMBEDDINGS",
                }
            )
    with gzip.open(root / "per_query.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in per:
            f.write(json.dumps(r) + "\n")
    summary = {k: {m: float(np.mean([r[m] for r in rows])) for m in rows[0]} for k, rows in groups.items()}
    dump(
        root / "summary.json",
        {
            "sources": {
                str(p): sha(p)
                for p in [
                    source,
                    old / "corpus_embeddings.npy",
                    old / "query_embeddings.npy",
                    Path("data/beir/scifact/qrels/test.tsv"),
                ]
            },
            "stored_branch_lengths": {
                s: sorted({len(r[s]) for r in archived.values()}) for s in ("BM25", "Dense_Exact")
            },
            "reconstructed_top50_mismatch_queries": mismatch,
            "metrics": summary,
            "finding": "The old registry re-fused only 50 saved hits per branch, called this D50/top_k=50, and advertised 250 branch candidates. Actual Stage 2 top_k=50 requests 250 candidates. Returned cutoff alone cannot change the first 10 ranks. Do not reconstruct a 250-depth run from 50-depth lists.",
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
