"""
RAGScaleBench Corpus Preparation Pipeline.
Performs schema validation, UTF-8 normalisation, language filtering, deduplication,
licence validation, deterministic sampling, and canonical Parquet/manifest generation.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import pandas as pd

from ragbench.config import load_config
from ragbench.data.arxiv import download_arxiv_records
from ragbench.data.govinfo import download_govinfo_records
from ragbench.data.wikipedia import download_wikipedia_records
from ragbench.ids import compute_content_sha256
from ragbench.provenance import (
    CanonicalDocument,
    CorpusManifest,
    validate_licence,
    write_manifest_lock,
)


def load_raw_jsonl(dir_path: str) -> list[CanonicalDocument]:
    """Load and validate all JSONL files in a directory."""
    docs: list[CanonicalDocument] = []
    if not os.path.exists(dir_path):
        return docs

    for fname in os.listdir(dir_path):
        if fname.endswith(".jsonl"):
            fpath = os.path.join(dir_path, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                        doc = CanonicalDocument(**raw)
                        docs.append(doc)
                    except Exception as e:
                        print(f"Skipping malformed record in {fpath}: {e}", file=sys.stderr)
    return docs


def prepare_corpus(
    config_path: str = "configs/default.yaml",
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    manifest_path: str = "data/data_manifest.lock.yaml",
    max_records: int | None = None,
) -> CorpusManifest:
    """Execute complete corpus normalisation and deduplication pipeline."""
    config = load_config(config_path)
    os.makedirs(processed_dir, exist_ok=True)

    arxiv_dir = os.path.join(raw_dir, "arxiv")
    wiki_dir = os.path.join(raw_dir, "wikipedia")
    gov_dir = os.path.join(raw_dir, "govinfo")

    target_arxiv = (
        config.arxiv_count if max_records is None else min(config.arxiv_count, max_records // 2)
    )
    target_wiki = (
        config.wikipedia_count
        if max_records is None
        else min(config.wikipedia_count, max_records // 2)
    )
    target_gov = (
        config.govinfo_count if max_records is None else min(config.govinfo_count, max_records // 4)
    )

    # 1. Acquire raw data if not present
    arxiv_docs = load_raw_jsonl(arxiv_dir)
    if len(arxiv_docs) < target_arxiv:
        print(f"Acquiring {target_arxiv} arXiv records...")
        download_arxiv_records(limit=target_arxiv, output_dir=arxiv_dir)
        arxiv_docs = load_raw_jsonl(arxiv_dir)

    wiki_docs = load_raw_jsonl(wiki_dir)
    if len(wiki_docs) < target_wiki:
        print(f"Acquiring {target_wiki} Wikipedia records...")
        download_wikipedia_records(limit=target_wiki, output_dir=wiki_dir)
        wiki_docs = load_raw_jsonl(wiki_dir)

    gov_docs = load_raw_jsonl(gov_dir)
    if len(gov_docs) < target_gov:
        print(f"Acquiring {target_gov} GovInfo records...")
        download_govinfo_records(limit=target_gov, output_dir=gov_dir)
        gov_docs = load_raw_jsonl(gov_dir)

    all_raw = arxiv_docs[:target_arxiv] + wiki_docs[:target_wiki] + gov_docs[:target_gov]

    # 2. Validation, UTF-8 Normalisation, Language checks, Deduplication
    seen_hashes: set[str] = set()
    cleaned_docs: list[CanonicalDocument] = []
    source_counts: dict[str, int] = defaultdict(int)
    source_snapshots: dict[str, str] = {}
    licence_decls: dict[str, str] = {}

    for doc in all_raw:
        # Schema & UTF-8 normalisation
        clean_text = doc.text.encode("utf-8", "ignore").decode("utf-8").strip()
        if len(clean_text) < 50:
            continue

        if doc.language.lower() != "en":
            continue

        valid, reason = validate_licence(doc.source, doc.licence_status)
        if not valid:
            continue

        content_sha = compute_content_sha256(clean_text)
        if content_sha in seen_hashes:
            continue

        seen_hashes.add(content_sha)
        doc.text = clean_text
        doc.content_sha256 = content_sha
        cleaned_docs.append(doc)

        source_counts[doc.source] += 1
        source_snapshots[doc.source] = doc.snapshot_id
        licence_decls[doc.source] = doc.licence

    # 3. Sort deterministically by doc_id
    cleaned_docs.sort(key=lambda d: d.doc_id)

    # 4. Output to Parquet and JSONL
    parquet_path = os.path.join(processed_dir, "documents.parquet")
    jsonl_path = os.path.join(processed_dir, "documents.jsonl")

    records = [d.model_dump() for d in cleaned_docs]
    df = pd.DataFrame(records)
    df.to_parquet(parquet_path, index=False, engine="pyarrow")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.writelines(d.model_dump_json() + "\n" for d in cleaned_docs)

    # 5. Compute total corpus checksum
    corpus_hasher = compute_content_sha256("".join(d.content_sha256 for d in cleaned_docs))
    manifest = CorpusManifest(
        total_documents=len(cleaned_docs),
        corpus_sha256=corpus_hasher,
        source_counts=dict(source_counts),
        source_snapshots=source_snapshots,
        licence_declarations=licence_decls,
    )
    write_manifest_lock(manifest, manifest_path)
    print(f"Corpus prepared successfully: {len(cleaned_docs)} documents saved to {parquet_path}")
    print(f"Manifest lock written to {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare canonical RAGScaleBench corpus.")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to default.yaml"
    )
    parser.add_argument(
        "--max-records", type=int, default=None, help="Optional ceiling for quick/smoke runs"
    )
    args = parser.parse_args()

    prepare_corpus(config_path=args.config, max_records=args.max_records)


if __name__ == "__main__":
    main()
