"""Independent BEIR audit. Run from repository root with python -m scripts.full_beir_reevaluation."""

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import time
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("results/recomputed")
MD5 = {
    "scifact": "5f7d1de60b170fc8027bb7898e2efca1",
    "nfcorpus": "a89dba18a62ef92f7d323ec890a0d38d",
    "fiqa": "17918ed23cd04fb15047f73e6c3bd9d9",
}
# Pinned from TLS-verified official archive downloads on 2026-10-24.
# BEIR publishes MD5, not these SHA-256 digests.
SHA256 = {
    "scifact": "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
    "nfcorpus": "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
    "fiqa": "32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02",
}


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(name):
    root = Path("data/beir") / name
    corpus = [json.loads(s) for s in (root / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    queries = [json.loads(s) for s in (root / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len({x["_id"] for x in corpus}) == len(corpus)
    assert len({x["_id"] for x in queries}) == len(queries)
    qrels = {}
    for p in sorted((root / "qrels").glob("*.tsv")):
        rows = defaultdict(dict)
        for r in csv.DictReader(p.open(encoding="utf-8"), delimiter="\t"):
            q, d, g = r["query-id"], r["corpus-id"], int(r["score"])
            assert d not in rows[q], (name, q, d)
            rows[q][d] = g
        qrels[p.stem] = dict(rows)
    return corpus, {r["_id"]: r["text"] for r in queries}, qrels


def verify():
    manifests = {}
    for name, expected in MD5.items():
        path = Path("data/beir") / f"{name}.zip"
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
        # Reacquire official bytes with TLS verification; preserve local archive for comparison.
        fresh = OUT / "downloads" / f"{name}.zip"
        fresh.parent.mkdir(exist_ok=True, parents=True)
        acquisition = {}
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                fresh.write_bytes(response.read())
            assert sha(fresh) == SHA256[name], "Downloaded archive SHA-256 mismatch"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(fresh.read_bytes())
            acquisition = {
                "status": "DOWNLOADED_TLS_VERIFIED",
                "sha256": sha(fresh),
                "matches_local": sha(fresh) == sha(path),
            }
        except Exception as exc:
            acquisition = {"status": "DOWNLOAD_BLOCKED", "error": str(exc)}
        assert hashlib.md5(path.read_bytes()).hexdigest() == expected, name
        assert sha(path) == SHA256[name], name
        files = {}
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if member.endswith((".jsonl", ".tsv")):
                    disk = Path("data/beir") / member
                    assert disk.resolve().is_relative_to(Path("data/beir").resolve()), "Unsafe archive path"
                    digest = hashlib.sha256(archive.read(member)).hexdigest()
                    if not disk.exists():
                        disk.parent.mkdir(parents=True, exist_ok=True)
                        disk.write_bytes(archive.read(member))
                    assert sha(disk) == digest, member
                    files[member] = digest
        corpus, queries, splits = load_data(name)
        cids = {r["_id"] for r in corpus}
        for rows in splits.values():
            assert set(rows) <= set(queries)
            assert all(set(ds) <= cids for ds in rows.values())
        overlaps = {}
        for a, b in itertools.combinations(splits, 2):
            ta = {" ".join(queries[q].lower().split()) for q in splits[a]}
            tb = {" ".join(queries[q].lower().split()) for q in splits[b]}
            overlaps[a + "__" + b] = {
                "query_ids": sorted(set(splits[a]) & set(splits[b])),
                "normalized_query_texts": sorted(ta & tb),
            }
        manifests[name] = {
            "archive_sha256": sha(path),
            "archive_md5": expected,
            "upstream_checksum_type": "MD5; SHA256 is locally computed, not an upstream published digest",
            "url": url,
            "acquisition": acquisition,
            "files_sha256": files,
            "documents": len(corpus),
            "queries": len(queries),
            "splits": {
                s: {
                    "queries": len(q),
                    "judgments": sum(map(len, q.values())),
                    "grades": sorted({g for ds in q.values() for g in ds.values()}),
                }
                for s, q in splits.items()
            },
            "overlap": overlaps,
            "license_card": f"https://huggingface.co/datasets/BeIR/{name}",
            "license_card_declares": "CC-BY-SA-4.0",
        }
        print(name, manifests[name]["splits"], acquisition, flush=True)
        dump(OUT / "dataset_verification.json", manifests)


def metrics(ids, gold):
    assert len(ids) == len(set(ids)), "duplicate retrieved document"
    relevant = {d for d, g in gold.items() if g > 0}
    result = {}
    for k in (1, 5, 10):
        hits = len(set(ids[:k]) & relevant)
        result[f"Recall@{k}"] = hits / len(relevant) if relevant else 0.0
        result[f"Precision@{k}"] = hits / k
    dcg = sum(max(gold.get(d, 0), 0) / math.log2(i + 2) for i, d in enumerate(ids[:10]))
    ideal = sum(
        g / math.log2(i + 2) for i, g in enumerate(sorted([g for g in gold.values() if g > 0], reverse=True)[:10])
    )
    binary = sum((d in relevant) / math.log2(i + 2) for i, d in enumerate(ids[:10]))
    bidcg = sum(1 / math.log2(i + 2) for i in range(min(10, len(relevant))))
    result["nDCG@10"] = dcg / ideal if ideal else 0.0
    result["binary_nDCG@10"] = binary / bidcg if bidcg else 0.0
    result["MRR@10"] = next((1 / (i + 1) for i, d in enumerate(ids[:10]) if d in relevant), 0.0)
    result["MRR@250"] = next((1 / (i + 1) for i, d in enumerate(ids[:250]) if d in relevant), 0.0)
    return result


def statistics(records):
    from ragbench.evaluation.bootstrap import adjust_p_values_bh, adjust_p_values_holm

    groups = defaultdict(dict)
    for r in records:
        groups[(r["depth"], r["system"])][r["query_id"]] = r["metrics"]
    results = []
    for depth in sorted({k[0] for k in groups}):
        systems = sorted(s for d, s in groups if d == depth)
        for a, b in itertools.combinations(systems, 2):
            ga, gb = groups[depth, a], groups[depth, b]
            assert ga.keys() == gb.keys()
            for metric in ("nDCG@10", "Recall@10"):
                diff = np.array([ga[q][metric] - gb[q][metric] for q in sorted(ga)])
                rng = np.random.default_rng(42)
                boots = []
                null = []
                for _ in range(100):
                    ix = rng.integers(0, len(diff), size=(100, len(diff)))
                    boots.extend(diff[ix].mean(axis=1))
                    null.extend(((diff - diff.mean())[ix]).mean(axis=1))
                low, high = np.quantile(boots, [0.025, 0.975])
                p = (1 + np.sum(np.abs(null) >= abs(diff.mean()))) / 10001
                results.append(
                    {
                        "depth": depth,
                        "a": a,
                        "b": b,
                        "metric": metric,
                        "n": len(diff),
                        "delta_a_minus_b": float(diff.mean()),
                        "ci95": [float(low), float(high)],
                        "p_centered_bootstrap": float(p),
                        "resamples": 10000,
                        "seed": 42,
                    }
                )
    # One conservative family: all pairwise tests, both metrics, all depths in this dataset.
    for r, h, b in zip(
        results,
        adjust_p_values_holm([r["p_centered_bootstrap"] for r in results]),
        adjust_p_values_bh([r["p_centered_bootstrap"] for r in results]),
    ):
        r.update(p_holm=h, q_bh=b, family="all systems x depths x two metrics within dataset")
    return results


def run(name):
    import pickle

    import psutil
    import torch
    from sentence_transformers import SentenceTransformer

    from ragbench.evaluation.bm25 import BM25Okapi
    from ragbench.retrieval.policies import AdaptiveRouterPolicy, AdaptiveThresholds, FixedHybridPolicy

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    np.random.seed(42)
    torch.manual_seed(42)
    root = OUT / name
    root.mkdir(exist_ok=True, parents=True)
    corpus, queries, splits = load_data(name)
    qrels = splits["test"]
    qids = sorted(qrels)
    cids = [r["_id"] for r in corpus]
    texts = [(r.get("title", "") + " " + r["text"]).strip() for r in corpus]
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    revision = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    model = SentenceTransformer(model_id, revision=revision, device="cpu", local_files_only=True)
    print(name, "rebuilding embeddings", len(cids), flush=True)
    start = time.perf_counter()
    vectors = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    build = time.perf_counter() - start
    np.save(root / "corpus_embeddings.npy", vectors)
    start = time.perf_counter()
    qvectors = model.encode(
        [queries[q] for q in qids], batch_size=32, normalize_embeddings=True, show_progress_bar=False
    )
    query_build = time.perf_counter() - start
    np.save(root / "query_embeddings.npy", qvectors)
    dump(root / "corpus_ids.json", cids)
    dump(root / "query_ids.json", qids)
    print(name, "embedding complete", build, flush=True)
    start = time.perf_counter()
    bm = BM25Okapi()
    bm.fit(list(zip(cids, texts)))
    bm_build = time.perf_counter() - start
    with (root / "bm25_index.pkl").open("wb") as f:
        pickle.dump(bm, f)

    def dense(vec, k):
        scores = vectors @ vec
        order = np.argsort(-scores, kind="stable")[:k]
        return [(cids[i], float(scores[i])) for i in order]

    records = []
    summaries = []
    # Fixed defaults avoid any test-set threshold selection. Archived calibrated configurations are separate.
    for depth in (50, 100, 250):
        hybrid = FixedHybridPolicy(bm, dense, candidate_depth=depth)
        router = AdaptiveRouterPolicy(bm, dense, thresholds=AdaptiveThresholds(), candidate_depth=depth)
        lat = defaultdict(list)
        bysys = defaultdict(list)
        for qi, qid in enumerate(qids):
            text = queries[qid]
            vec = qvectors[qi]
            for system in ("BM25", "Dense_Exact", "Fixed_Hybrid", "Adaptive_Router"):
                start = time.perf_counter()
                route = system
                if system == "BM25":
                    hits = bm.search(text, top_k=depth)
                elif system == "Dense_Exact":
                    hits = dense(vec, depth)
                else:
                    res = (hybrid if system == "Fixed_Hybrid" else router).retrieve(text, vec, top_k=depth)
                    hits = res.retrieved_docs
                    route = res.route_taken
                ms = (time.perf_counter() - start) * 1000
                score = metrics([d for d, s in hits], qrels[qid])
                # MRR has a depth-specific censoring limit.
                score[f"MRR@{depth}"] = score.pop("MRR@250")
                r = {
                    "query_id": qid,
                    "system": system,
                    "depth": depth,
                    "route": route,
                    "latency_ms": ms,
                    "rankings": hits,
                    "metrics": score,
                }
                records.append(r)
                bysys[system].append(r)
                lat[system].append(ms)
        for system, rs in bysys.items():
            summaries.append(
                {
                    "dataset": name,
                    "system": system,
                    "candidate_depth_per_branch": depth,
                    "returned_depth": depth,
                    "model": model_id if system != "BM25" else "BM25 k1=1.5 b=0.75",
                    "metrics": {k: float(np.mean([r["metrics"][k] for r in rs])) for k in rs[0]["metrics"]},
                    "p50_ms": float(np.percentile(lat[system], 50)),
                    "p95_ms": float(np.percentile(lat[system], 95)),
                    "p99_ms": float(np.percentile(lat[system], 99)),
                    "serial_search_qps": 1000 / float(np.mean(lat[system])),
                    "latency_scope": "single pass serial warm retrieval; precomputed query embeddings; excludes generation/network/encoding; not service QPS",
                    "process_rss_bytes": psutil.Process().memory_info().rss,
                    "lexical_routes": sum(r["route"] == "bm25" for r in rs),
                    "n": len(rs),
                    "evidence": "NEW_EXECUTION",
                }
            )
        print(name, "depth", depth, "complete", flush=True)
    with gzip.open(root / "per_query.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    dump(root / "metrics.json", summaries)
    dump(root / "statistics.json", statistics(records))
    legacy = Path("results") / name / "corpus_embeddings.npy"
    comparison = None
    if legacy.exists():
        old = np.load(legacy)
        oldids = json.loads((legacy.parent / "corpus_ids.json").read_text())
        comparison = {"ids_equal": oldids == cids, "shape_equal": old.shape == vectors.shape}
        if oldids == cids and old.shape == vectors.shape:
            comparison["max_abs_difference"] = float(np.max(np.abs(old - vectors)))
    dump(
        root / "build.json",
        {
            "model": model_id,
            "revision": revision,
            "max_seq_length": model.max_seq_length,
            "device": "cpu",
            "threads": 2,
            "seed": 42,
            "corpus_encoding_seconds": build,
            "query_encoding_seconds": query_build,
            "bm25_build_seconds": bm_build,
            "corpus_vectors_bytes": (root / "corpus_embeddings.npy").stat().st_size,
            "bm25_pickle_bytes": (root / "bm25_index.pkl").stat().st_size,
            "corpus_vectors_sha256": sha(root / "corpus_embeddings.npy"),
            "query_vectors_sha256": sha(root / "query_embeddings.npy"),
            "old_embedding_comparison": comparison,
            "thresholds": vars(AdaptiveThresholds()),
            "process_peak_wset_bytes": getattr(psutil.Process().memory_info(), "peak_wset", None),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    p.add_argument("--dataset", choices=list(MD5))
    args = p.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    if args.verify:
        verify()
    if args.dataset:
        run(args.dataset)
