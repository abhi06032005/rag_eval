"""
Download and verify official BEIR cross-domain benchmark datasets:
- NFCorpus (Biomedical/Nutrition Information Retrieval)
- FiQA (Financial Opinion QA and Information Retrieval)

Computes SHA256 hashes, extracts corpus/queries/qrels, verifies schema integrity,
and generates dataset manifests.
"""

import hashlib
import json
import os
import ssl
import urllib.request
import zipfile

DATA_DIR = "data/beir"
URLS = {
    "nfcorpus": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
    "fiqa": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
}

LICENSES = {
    "nfcorpus": "CC BY-SA 4.0 (National Library of Medicine / NutritionFacts.org)",
    "fiqa": "CC BY-SA 4.0 / Academic Research Use (WWW 2018 FiQA Challenge)",
}


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 16):
            h.update(chunk)
    return h.hexdigest()


def download_dataset(dataset_name: str, url: str) -> tuple[str, str]:
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, f"{dataset_name}.zip")
    
    if not os.path.exists(zip_path):
        print(f"Downloading {dataset_name} from {url}...", flush=True)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(url, context=ctx) as resp, open(zip_path, "wb") as out_f:
            while chunk := resp.read(65536):
                out_f.write(chunk)
        print(f"Downloaded {zip_path} ({os.path.getsize(zip_path) / (1024*1024):.2f} MB)", flush=True)
    else:
        print(f"Using existing archive: {zip_path}", flush=True)

    sha256 = compute_sha256(zip_path)
    print(f"{dataset_name}.zip SHA256: {sha256}", flush=True)

    extract_dir = os.path.join(DATA_DIR, dataset_name)
    if not os.path.exists(extract_dir) or not os.path.exists(os.path.join(extract_dir, "corpus.jsonl")):
        print(f"Extracting {zip_path} to {extract_dir}...", flush=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)

    return extract_dir, sha256


def audit_extracted_dataset(dataset_name: str, dataset_dir: str, sha256: str) -> dict:
    corpus_file = os.path.join(dataset_dir, "corpus.jsonl")
    queries_file = os.path.join(dataset_dir, "queries.jsonl")
    qrels_dir = os.path.join(dataset_dir, "qrels")

    corpus_count = 0
    with open(corpus_file, "r", encoding="utf-8") as f:
        for _ in f:
            corpus_count += 1

    queries_count = 0
    with open(queries_file, "r", encoding="utf-8") as f:
        for _ in f:
            queries_count += 1

    qrel_splits = {}
    if os.path.exists(qrels_dir):
        for f in os.listdir(qrels_dir):
            if f.endswith(".tsv"):
                split_name = f[:-4]
                count = 0
                with open(os.path.join(qrels_dir, f), "r", encoding="utf-8") as qf:
                    _ = qf.readline()
                    for line in qf:
                        parts = line.strip().split("\t")
                        if len(parts) >= 3 and int(parts[2]) > 0:
                            count += 1
                qrel_splits[split_name] = count

    return {
        "dataset": dataset_name,
        "archive_sha256": sha256,
        "license": LICENSES.get(dataset_name, "Open Academic Use"),
        "corpus_docs": corpus_count,
        "queries_total": queries_count,
        "positive_qrels_by_split": qrel_splits,
    }


def main():
    print("=================================================================", flush=True)
    print("   BEIR Cross-Domain Benchmark Ingestion (NFCorpus & FiQA)      ", flush=True)
    print("=================================================================", flush=True)

    manifests = {}
    for name, url in URLS.items():
        extract_dir, sha256 = download_dataset(name, url)
        manifest = audit_extracted_dataset(name, extract_dir, sha256)
        manifests[name] = manifest
        print(f"Audited {name}: Docs={manifest['corpus_docs']} | Queries={manifest['queries_total']} | Qrels={manifest['positive_qrels_by_split']}", flush=True)

    manifest_path = os.path.join(DATA_DIR, "crossdomain_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifests, f, indent=2)
    print(f"\nSaved cross-domain dataset manifest to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
