"""Real HTTP Qdrant HNSW/quantization runs on rebuilt vectors; local emulation is forbidden."""

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import psutil
from qdrant_client import QdrantClient
from qdrant_client import models as m

from scripts.full_beir_reevaluation import OUT, dump, load_data, metrics


def main(name):
    root = OUT / name
    out = root / "qdrant"
    out.mkdir(exist_ok=True)
    vec = np.load(root / "corpus_embeddings.npy")
    qvec = np.load(root / "query_embeddings.npy")
    cids = json.loads((root / "corpus_ids.json").read_text())
    qids = json.loads((root / "query_ids.json").read_text())
    gold = load_data(name)[2]["test"]
    client = QdrantClient(url="http://127.0.0.1:6335", timeout=120)
    dump(out / "server.json", client.info().model_dump(mode="json"))
    exact = {q: set(np.argsort(-(vec @ v), kind="stable")[:10].tolist()) for q, v in zip(qids, qvec)}
    configs = [
        ("hnsw", 16, 100, None),
        ("hnsw_tuned", 32, 200, None),
        (
            "int8",
            32,
            200,
            m.ScalarQuantization(
                scalar=m.ScalarQuantizationConfig(type=m.ScalarType.INT8, quantile=0.99, always_ram=True)
            ),
        ),
        ("binary", 32, 200, m.BinaryQuantization(binary=m.BinaryQuantizationConfig(always_ram=True))),
    ]
    summary = []
    records = []
    for label, degree, efc, quant in configs:
        collection = f"reevaluation_{name}_{label}"
        if client.collection_exists(collection):
            raise RuntimeError("Refusing to overwrite existing collection " + collection)
        start = time.perf_counter()
        client.create_collection(
            collection,
            vectors_config=m.VectorParams(size=vec.shape[1], distance=m.Distance.COSINE),
            hnsw_config=m.HnswConfigDiff(m=degree, ef_construct=efc, full_scan_threshold=10, max_indexing_threads=2),
            optimizers_config=m.OptimizersConfigDiff(
                indexing_threshold=10, default_segment_number=1, max_optimization_threads=1
            ),
            quantization_config=quant,
        )
        for off in range(0, len(vec), 256):
            client.upsert(
                collection,
                points=m.Batch(ids=list(range(off, min(off + 256, len(vec)))), vectors=vec[off : off + 256].tolist()),
                wait=True,
            )
        deadline = time.monotonic() + 900
        while True:
            info = client.get_collection(collection)
            if info.status == m.CollectionStatus.GREEN and info.indexed_vectors_count == len(vec):
                break
            if time.monotonic() > deadline:
                dump(
                    out / (label + "-blocked.json"),
                    {
                        "status": "BLOCKED",
                        "reason": "Full HNSW indexing not confirmed before 900s timeout",
                        "info": info.model_dump(mode="json"),
                    },
                )
                break
            time.sleep(2)
        if info.indexed_vectors_count != len(vec):
            continue
        build = time.perf_counter() - start
        dump(out / (label + "-collection.json"), info.model_dump(mode="json"))
        for ef in [32, 128] if label in ("hnsw", "hnsw_tuned") else [128]:
            for rescore in [False, True] if quant else [False]:
                cfg = f"{label}_ef{ef}_rescore{rescore}"
                rows = []
                params = m.SearchParams(
                    hnsw_ef=ef,
                    exact=False,
                    quantization=m.QuantizationSearchParams(ignore=False, rescore=rescore, oversampling=2.0)
                    if quant
                    else None,
                )
                client.query_points(
                    collection, query=qvec[0].tolist(), limit=10, search_params=params, with_payload=False
                )
                for q, v in zip(qids, qvec):
                    start = time.perf_counter()
                    hits = client.query_points(
                        collection, query=v.tolist(), limit=10, search_params=params, with_payload=False
                    ).points
                    ms = (time.perf_counter() - start) * 1000
                    ids = [cids[int(p.id)] for p in hits]
                    score = metrics(ids, gold[q])
                    score.pop("MRR@250")
                    r = {
                        "query_id": q,
                        "configuration": cfg,
                        "rankings": [(cids[int(p.id)], p.score) for p in hits],
                        "metrics": score,
                        "ann_recall10": len({int(p.id) for p in hits} & exact[q]) / 10,
                        "latency_ms": ms,
                    }
                    rows.append(r)
                    records.append(r)
                directory = Path("storage/recomputed-audit/collections") / collection
                pid = int((OUT / "qdrant-pid.txt").read_text().strip())
                rss = psutil.Process(pid).memory_info().rss
                summary.append(
                    {
                        "dataset": name,
                        "configuration": cfg,
                        "metrics": {k: float(np.mean([r["metrics"][k] for r in rows])) for k in rows[0]["metrics"]},
                        "ANN_Recall@10": float(np.mean([r["ann_recall10"] for r in rows])),
                        "p50_ms": float(np.percentile([r["latency_ms"] for r in rows], 50)),
                        "p95_ms": float(np.percentile([r["latency_ms"] for r in rows], 95)),
                        "p99_ms": float(np.percentile([r["latency_ms"] for r in rows], 99)),
                        "serial_http_qps": 1000 / float(np.mean([r["latency_ms"] for r in rows])),
                        "collection_disk_bytes": sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()),
                        "server_process_rss_bytes": rss,
                        "memory_scope": "entire server with previously built collections still resident; not per-index memory",
                        "build_upload_and_optimize_seconds": build,
                        "indexed_vectors": info.indexed_vectors_count,
                        "n": len(rows),
                        "evidence": "NEW_REAL_QDRANT_HTTP",
                        "latency_scope": "one warm serial HTTP pass; precomputed query vectors; excludes embedding/generation",
                    }
                )
                dump(out / "metrics.json", summary)
                print(name, cfg, summary[-1]["ANN_Recall@10"], flush=True)
        with gzip.open(out / "per_query.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    main(p.parse_args().dataset)
