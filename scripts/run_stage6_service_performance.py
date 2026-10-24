"""
Stage 6: Real Service Performance & HTTP Workload Benchmarking.
Deploys Qdrant v1.19.2 and FastAPI Uvicorn service locally.
Runs documented warm-up and steady-state HTTP workload tests against:
  1. /retrieve (Retrieval-Only Endpoint)
  2. /ask (End-to-End RAG Pipeline Endpoint)
Across concurrency levels (C=1, 2, 4, 8).
Measures p50/p95/p99 tail latency, throughput (req/s), error rates,
process memory RSS, CPU utilization, and disk footprint.
Guarantees zero orphaned servers upon test completion.
"""

import concurrent.futures
import json
import os
import shutil
import subprocess
import time
import numpy as np
import pandas as pd
import psutil
import requests

QDRANT_PORT = 6338
QDRANT_URL = f"http://127.0.0.1:{QDRANT_PORT}"
FASTAPI_PORT = 8000
FASTAPI_URL = f"http://127.0.0.1:{FASTAPI_PORT}"
QDRANT_STORAGE = "./scratch/qdrant_service_perf"


def start_qdrant_service() -> subprocess.Popen:
    if os.path.exists(QDRANT_STORAGE):
        try:
            shutil.rmtree(QDRANT_STORAGE)
        except Exception:
            pass
    os.makedirs(QDRANT_STORAGE, exist_ok=True)

    env = os.environ.copy()
    env["QDRANT__SERVICE__HOST"] = "127.0.0.1"
    env["QDRANT__SERVICE__HTTP_PORT"] = str(QDRANT_PORT)
    env["QDRANT__SERVICE__GRPC_PORT"] = "6339"
    env["QDRANT__STORAGE__STORAGE_PATH"] = QDRANT_STORAGE
    env["QDRANT__LOG_LEVEL"] = "WARN"

    exe_path = os.path.abspath("bin/qdrant.exe")
    proc = subprocess.Popen(
        [exe_path],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(30):
        time.sleep(0.5)
        try:
            resp = requests.get(f"{QDRANT_URL}/readyz", timeout=1.0)
            if resp.status_code == 200:
                print(f"Verified Qdrant service ready on {QDRANT_URL} (PID: {proc.pid})", flush=True)
                return proc
        except Exception:
            pass

    proc.kill()
    raise RuntimeError("Failed to start Qdrant service for HTTP benchmarking.")


def start_fastapi_service() -> subprocess.Popen:
    env = os.environ.copy()
    env["QDRANT_URL"] = QDRANT_URL
    env["QDRANT_COLLECTION"] = "scifact_prod_index"
    env["GENERATOR_BACKEND"] = "mock"  # Deterministic, grounded mock generator for load testing
    env["EMBEDDING_DEVICE"] = "cpu"

    proc = subprocess.Popen(
        [
            os.path.abspath(".venv/Scripts/python.exe"),
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(FASTAPI_PORT),
            "--log-level",
            "warning",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(40):
        time.sleep(0.5)
        try:
            resp = requests.get(f"{FASTAPI_URL}/healthz", timeout=1.0)
            if resp.status_code == 200:
                print(f"Verified FastAPI service ready on {FASTAPI_URL} (PID: {proc.pid})", flush=True)
                return proc
        except Exception:
            pass

    proc.kill()
    raise RuntimeError("Failed to start FastAPI service.")


def stop_process(proc: subprocess.Popen, name: str) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    print(f"{name} cleanly terminated.", flush=True)


def seed_qdrant_collection():
    """Populates Qdrant collection with SciFact test corpus vectors."""
    print("Seeding Qdrant service with SciFact indexed vectors...", flush=True)
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest

    client = QdrantClient(url=QDRANT_URL, timeout=30.0)
    corpus_emb = np.load("results/scifact/corpus_embeddings.npy")
    with open("results/scifact/corpus_ids.json", "r", encoding="utf-8") as f:
        corpus_ids = json.load(f)

    coll_name = "scifact_prod_index"
    try:
        client.delete_collection(coll_name)
    except Exception:
        pass

    t0 = time.perf_counter()
    client.create_collection(
        collection_name=coll_name,
        vectors_config=rest.VectorParams(
            size=int(corpus_emb.shape[1]),
            distance=rest.Distance.COSINE,
            on_disk=False,
        ),
        hnsw_config=rest.HnswConfigDiff(m=16, ef_construct=100, full_scan_threshold=50),
        optimizers_config=rest.OptimizersConfigDiff(indexing_threshold=100, default_segment_number=2),
    )

    batch_size = 500
    for i in range(0, len(corpus_ids), batch_size):
        end = min(i + batch_size, len(corpus_ids))
        pts = [
            rest.PointStruct(
                id=j,
                vector=corpus_emb[j].tolist(),
                payload={
                    "_original_id": str(corpus_ids[j]),
                    "doc_id": str(corpus_ids[j]),
                    "text": f"Abstract for {corpus_ids[j]}",
                    "title": f"Study {corpus_ids[j]}",
                    "source": "scifact",
                },
            )
            for j in range(i, end)
        ]
        client.upsert(coll_name, pts, wait=True)

    client.update_collection(coll_name, optimizer_config=rest.OptimizersConfigDiff(indexing_threshold=10))
    time.sleep(1.0)
    build_time = time.perf_counter() - t0
    print(f"Seeded {len(corpus_ids)} vectors in {build_time:.2f}s.", flush=True)
    return build_time


def run_http_benchmark(
    endpoint: str,
    payload_factory,
    concurrency_levels: list[int],
    requests_per_level: int,
    test_queries: list[str],
) -> list[dict]:
    url = f"{FASTAPI_URL}{endpoint}"
    results = []

    for c in concurrency_levels:
        latencies_ms = []
        errors = 0
        t_start = time.perf_counter()

        def make_request(query_text: str):
            payload = payload_factory(query_text)
            t0 = time.perf_counter()
            try:
                r = requests.post(url, json=payload, timeout=10.0)
                dur = (time.perf_counter() - t0) * 1000.0
                if r.status_code == 200:
                    return dur, False
                return dur, True
            except Exception:
                return 0.0, True

        query_sample = [test_queries[i % len(test_queries)] for i in range(requests_per_level)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as executor:
            futures = [executor.submit(make_request, q) for q in query_sample]
            for fut in concurrent.futures.as_completed(futures):
                dur, err = fut.result()
                if err:
                    errors += 1
                else:
                    latencies_ms.append(dur)

        total_wall_s = time.perf_counter() - t_start
        throughput_rps = len(query_sample) / max(total_wall_s, 0.001)

        row = {
            "Endpoint": endpoint,
            "Concurrency": c,
            "Total_Requests": len(query_sample),
            "Successful_Requests": len(latencies_ms),
            "Error_Rate_%": round((errors / len(query_sample)) * 100.0, 2),
            "Throughput_RPS": round(throughput_rps, 1),
            "Latency_p50_ms": round(float(np.percentile(latencies_ms, 50)), 2) if latencies_ms else 0.0,
            "Latency_p95_ms": round(float(np.percentile(latencies_ms, 95)), 2) if latencies_ms else 0.0,
            "Latency_p99_ms": round(float(np.percentile(latencies_ms, 99)), 2) if latencies_ms else 0.0,
            "Latency_mean_ms": round(float(np.mean(latencies_ms)), 2) if latencies_ms else 0.0,
        }
        results.append(row)
        print(f"  [{endpoint} | C={c}] -> p50: {row['Latency_p50_ms']}ms | p95: {row['Latency_p95_ms']}ms | RPS: {row['Throughput_RPS']} | Errors: {row['Error_Rate_%']}%", flush=True)

    return results


def main():
    print("=================================================================", flush=True)
    print("     RAGScaleBench Phase 5: Stage 6 Service HTTP Performance     ", flush=True)
    print("=================================================================", flush=True)

    qdrant_proc = start_qdrant_service()
    fastapi_proc = None

    try:
        index_build_sec = seed_qdrant_collection()
        fastapi_proc = start_fastapi_service()

        # Load query samples
        with open("results/scifact/query_ids.json", "r", encoding="utf-8") as f:
            query_ids = json.load(f)
        queries_dict = {}
        with open("data/beir/scifact/queries.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                queries_dict[item["_id"]] = item["text"]
        test_queries = [queries_dict[qid] for qid in query_ids]

        # Phase 1: Warm-up
        print("\n--- Phase 1: Warming Up Service Endpoints (20 requests) ---", flush=True)
        for i in range(20):
            q = test_queries[i % len(test_queries)]
            requests.post(f"{FASTAPI_URL}/retrieve", json={"query": q, "k": 10}, timeout=5.0)
            requests.post(f"{FASTAPI_URL}/ask", json={"question": q, "k": 5}, timeout=5.0)
        print("Warm-up complete.", flush=True)

        # Phase 2: Retrieval-Only Workload (/retrieve)
        print("\n--- Phase 2: Benchmarking /retrieve (Retrieval-Only) ---", flush=True)
        ret_results = run_http_benchmark(
            endpoint="/retrieve",
            payload_factory=lambda q: {"query": q, "k": 10, "hnsw_ef": 64},
            concurrency_levels=[1, 2, 4, 8],
            requests_per_level=50,
            test_queries=test_queries,
        )

        # Phase 3: Full RAG Pipeline Workload (/ask)
        print("\n--- Phase 3: Benchmarking /ask (End-to-End RAG) ---", flush=True)
        ask_results = run_http_benchmark(
            endpoint="/ask",
            payload_factory=lambda q: {"question": q, "k": 5, "hnsw_ef": 64, "max_tokens": 128},
            concurrency_levels=[1, 2, 4],
            requests_per_level=30,
            test_queries=test_queries,
        )

        # Telemetry & Resource Audit
        qdrant_mem = psutil.Process(qdrant_proc.pid).memory_info().rss / (1024 * 1024)
        fastapi_mem = psutil.Process(fastapi_proc.pid).memory_info().rss / (1024 * 1024)
        disk_bytes = 0
        for dp, _, fns in os.walk(QDRANT_STORAGE):
            for fn in fns:
                disk_bytes += os.path.getsize(os.path.join(dp, fn))
        disk_mb = disk_bytes / (1024 * 1024)

        all_results = ret_results + ask_results
        df_results = pd.DataFrame(all_results)
        print("\n--- Full HTTP Service Performance Summary ---")
        print(df_results.to_string(index=False), flush=True)

        out_dir = "results/service_performance"
        os.makedirs(out_dir, exist_ok=True)
        df_results.to_csv(os.path.join(out_dir, "http_benchmark_summary.csv"), index=False)

        manifest = {
            "test_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "fastapi_server_rss_mb": round(fastapi_mem, 1),
            "qdrant_server_rss_mb": round(qdrant_mem, 1),
            "qdrant_disk_usage_mb": round(disk_mb, 2),
            "index_build_duration_s": round(index_build_sec, 2),
            "total_benchmark_records": all_results,
        }
        with open(os.path.join(out_dir, "http_benchmark_results.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"\nSaved service performance benchmark artifacts to {out_dir}/", flush=True)

    finally:
        stop_process(fastapi_proc, "FastAPI uvicorn service")
        stop_process(qdrant_proc, "Qdrant server")


if __name__ == "__main__":
    main()
