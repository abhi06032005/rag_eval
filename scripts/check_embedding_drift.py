import json
from pathlib import Path

import numpy as np

from scripts.full_beir_reevaluation import load_data, metrics

for name in ("scifact", "nfcorpus"):
    old = Path("results") / name
    new = Path("results/recomputed") / name
    oi = json.loads((old / "query_ids.json").read_text())
    ni = json.loads((new / "query_ids.json").read_text())
    ov = np.load(old / "query_embeddings.npy")
    nv = np.load(new / "query_embeddings.npy")
    lookup = {q: i for i, q in enumerate(oi)}
    dif = np.array([np.max(np.abs(ov[lookup[q]] - v)) for q, v in zip(ni, nv)])
    cv = np.load(new / "corpus_embeddings.npy")
    ci = json.loads((new / "corpus_ids.json").read_text())
    gold = load_data(name)[2]["test"]
    sums = {}
    changed = []
    for q, v in zip(ni, nv):
        oldscores = cv @ ov[lookup[q]]
        newscores = cv @ v
        variants = {
            "archive_vectors_legacy_sort": np.argsort(oldscores)[::-1][:10],
            "new_vectors_legacy_sort": np.argsort(newscores)[::-1][:10],
            "new_vectors_stable_sort": np.argsort(-newscores, kind="stable")[:10],
        }
        if variants["archive_vectors_legacy_sort"].tolist() != variants["new_vectors_stable_sort"].tolist():
            changed.append(q)
        for label, inds in variants.items():
            sums.setdefault(label, []).append(metrics([ci[i] for i in inds], gold[q]))
    result = {
        "query_embedding_max_abs_difference": float(dif.max()),
        "queries_with_different_float_vectors": int((dif > 0).sum()),
        "exact_top10_changed_query_ids": changed,
        "metrics": {
            k: {m: float(np.mean([s[m] for s in rows])) for m in ["nDCG@10", "binary_nDCG@10", "Recall@10"]}
            for k, rows in sums.items()
        },
    }
    (new / "floating_point_drift.json").write_text(json.dumps(result, indent=2))
    print(name, result)
