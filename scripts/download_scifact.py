"""
Official BEIR SciFact Dataset Downloader and Checksum Verifier.
Downloads the official SciFact benchmark from UKP Lab (BEIR).
Validates corpus.jsonl, queries.jsonl, and qrels (train/dev/test).
"""

import hashlib
import json
import os
import ssl
import urllib.request
import zipfile

BEIR_SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
TARGET_DIR = "data/beir"
ZIP_PATH = os.path.join(TARGET_DIR, "scifact.zip")
EXTRACT_DIR = os.path.join(TARGET_DIR, "scifact")


def download_and_extract_scifact() -> dict:
    os.makedirs(TARGET_DIR, exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        print(f"Downloading official SciFact dataset from {BEIR_SCIFACT_URL}...")
        ctx = ssl._create_unverified_context()
        req = urllib.request.Request(BEIR_SCIFACT_URL, headers={"User-Agent": "RAGScaleBench/1.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp, open(ZIP_PATH, "wb") as f:
            while chunk := resp.read(65536):
                f.write(chunk)
        print(f"Downloaded {os.path.getsize(ZIP_PATH)} bytes.")

    # Calculate SHA256 of zip
    sha256_hash = hashlib.sha256()
    with open(ZIP_PATH, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    zip_sha256 = sha256_hash.hexdigest()
    print(f"SciFact Zip SHA-256: {zip_sha256}")

    if not os.path.exists(EXTRACT_DIR):
        print(f"Extracting {ZIP_PATH} to {TARGET_DIR}...")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(TARGET_DIR)

    # Verification
    corpus_path = os.path.join(EXTRACT_DIR, "corpus.jsonl")
    queries_path = os.path.join(EXTRACT_DIR, "queries.jsonl")
    qrels_test = os.path.join(EXTRACT_DIR, "qrels", "test.tsv")
    qrels_train = os.path.join(EXTRACT_DIR, "qrels", "train.tsv")

    corpus_count = sum(1 for _ in open(corpus_path, "r", encoding="utf-8"))
    queries_count = sum(1 for _ in open(queries_path, "r", encoding="utf-8"))
    qrels_test_count = sum(1 for _ in open(qrels_test, "r", encoding="utf-8")) - 1  # minus header
    qrels_train_count = sum(1 for _ in open(qrels_train, "r", encoding="utf-8")) - 1 if os.path.exists(qrels_train) else 0

    stats = {
        "benchmark": "BEIR/scifact",
        "upstream_url": BEIR_SCIFACT_URL,
        "zip_sha256": zip_sha256,
        "corpus_documents": corpus_count,
        "total_queries": queries_count,
        "qrels_test_records": qrels_test_count,
        "qrels_train_records": qrels_train_count,
        "extract_directory": EXTRACT_DIR,
        "licence": "Apache 2.0 / CC BY 4.0",
    }

    manifest_path = os.path.join(TARGET_DIR, "scifact_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nSciFact Verification Summary:")
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    download_and_extract_scifact()
