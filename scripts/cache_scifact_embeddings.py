"""
Precompute and Cache SciFact Dense Embeddings to Disk.
Runs sentence-transformers/all-MiniLM-L6-v2 with restricted CPU threads (2 threads)
to prevent machine heating, and saves exact embeddings to .npy for zero-overhead reuse.
"""

import json
import os
import time
import numpy as np
import torch

torch.set_num_threads(2)
from sentence_transformers import SentenceTransformer

def main():
    out_dir = "results/scifact"
    os.makedirs(out_dir, exist_ok=True)
    
    corpus_emb_path = os.path.join(out_dir, "corpus_embeddings.npy")
    corpus_ids_path = os.path.join(out_dir, "corpus_ids.json")
    query_emb_path = os.path.join(out_dir, "query_embeddings.npy")
    query_ids_path = os.path.join(out_dir, "query_ids.json")
    
    if os.path.exists(corpus_emb_path) and os.path.exists(query_emb_path):
        print(f"Embeddings already cached at {out_dir}. Skipping computation.")
        return

    print("Loading sentence-transformers/all-MiniLM-L6-v2 on CPU (threads=2)...")
    t0 = time.perf_counter()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s")
    
    # Load corpus
    print("Loading SciFact corpus...")
    corpus_ids = []
    corpus_texts = []
    with open("data/beir/scifact/corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            corpus_ids.append(item["_id"])
            full_text = (item.get("title", "") + " " + item.get("text", "")).strip()
            corpus_texts.append(full_text)
            
    print(f"Loaded {len(corpus_ids)} documents. Encoding in batches of 64...")
    t_enc_start = time.perf_counter()
    corpus_embeddings = model.encode(
        corpus_texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    enc_duration = time.perf_counter() - t_enc_start
    print(f"Corpus encoded in {enc_duration:.2f}s ({len(corpus_texts)/enc_duration:.1f} docs/sec)")
    
    np.save(corpus_emb_path, corpus_embeddings)
    with open(corpus_ids_path, "w", encoding="utf-8") as f:
        json.dump(corpus_ids, f)
    print(f"Saved corpus embeddings to {corpus_emb_path} (shape: {corpus_embeddings.shape})")

    # Load test queries
    print("Loading SciFact test queries with positive qrels...")
    test_qids = set()
    with open("data/beir/scifact/qrels/test.tsv", "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3 and int(parts[2]) > 0:
                test_qids.add(parts[0])
                
    query_ids = []
    query_texts = []
    with open("data/beir/scifact/queries.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["_id"] in test_qids:
                query_ids.append(item["_id"])
                query_texts.append(item["text"])
                
    print(f"Loaded {len(query_ids)} test queries. Encoding...")
    t_q_start = time.perf_counter()
    query_embeddings = model.encode(
        query_texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    q_dur = time.perf_counter() - t_q_start
    print(f"Queries encoded in {q_dur:.2f}s ({len(query_texts)/q_dur:.1f} queries/sec)")
    
    np.save(query_emb_path, query_embeddings)
    with open(query_ids_path, "w", encoding="utf-8") as f:
        json.dump(query_ids, f)
    print(f"Saved query embeddings to {query_emb_path} (shape: {query_embeddings.shape})")
    
    # Save encoding telemetry
    telemetry = {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": int(corpus_embeddings.shape[1]),
        "corpus_count": len(corpus_ids),
        "corpus_encoding_seconds": round(enc_duration, 2),
        "corpus_docs_per_sec": round(len(corpus_texts)/enc_duration, 1),
        "query_count": len(query_ids),
        "query_encoding_seconds": round(q_dur, 2),
        "threads": 2,
        "device": "cpu",
        "corpus_file_size_bytes": os.path.getsize(corpus_emb_path)
    }
    with open(os.path.join(out_dir, "embedding_telemetry.json"), "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)
    print("Telemetry saved.")

if __name__ == "__main__":
    main()
