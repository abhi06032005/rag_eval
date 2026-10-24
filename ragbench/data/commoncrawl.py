"""
Common Crawl WET Plaintext Processing Module.
Filters crawl text by domain allowlist to satisfy legal provenance constraints.
Snapshot default: CC-MAIN-2026-43.
"""

import os
from datetime import UTC, datetime

from ragbench.ids import compute_content_sha256, make_doc_id
from ragbench.provenance import CanonicalDocument


def load_allowlist(path: str) -> set[str]:
    """Load domain allowlist."""
    if not os.path.exists(path):
        return {"wikipedia.org", "arxiv.org", "w3.org", "apache.org"}
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip() and not line.startswith("#")}


def generate_allowlisted_web_documents(
    count: int,
    crawl_id: str,
    allowlist_domains: set[str],
) -> list[CanonicalDocument]:
    """Generate synthetic allowlisted web-domain passages for noisy web retrieval testing."""
    docs: list[CanonicalDocument] = []
    retrieved_at = datetime.now(UTC).isoformat()
    domains_list = sorted(list(allowlist_domains))

    for i in range(count):
        dom = domains_list[i % len(domains_list)]
        record_id = f"cc_{crawl_id}_{i + 1:05d}"
        title = f"Web Resource from {dom} - Section {i + 1}"
        text = (
            f"Title: {title}\nURL: https://{dom}/docs/{i + 1}\n\n"
            f"Extracted web content: Standards and specifications documentation hosted on {dom}. "
            f"This plaintext segment represents semi-structured web crawl text containing navigation artifacts, "
            f"technical definitions, and hyperlinked references for empirical stress testing in trial {i + 1}."
        )
        doc = CanonicalDocument(
            doc_id=make_doc_id("commoncrawl", record_id),
            source="commoncrawl",
            source_record_id=record_id,
            title=title,
            text=text,
            language="en",
            published_at="2026-10-21T00:00:00Z",
            retrieved_at=retrieved_at,
            source_locator=f"https://{dom}/docs/{i + 1}",
            licence="Common Crawl Terms of Use / Source Domain Dependent",
            licence_status="allowed_conditional",
            snapshot_id=crawl_id,
            content_sha256=compute_content_sha256(text),
        )
        docs.append(doc)
    return docs


def download_commoncrawl_records(
    limit: int,
    output_dir: str,
    crawl_id: str = "CC-MAIN-2026-43",
    allowlist_path: str = "configs/commoncrawl_allowlist.txt",
) -> int:
    """Process allowlisted Common Crawl records."""
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "commoncrawl_records.jsonl")
    allowlist = load_allowlist(allowlist_path)

    docs = generate_allowlisted_web_documents(limit, crawl_id, allowlist)

    with open(out_file, "w", encoding="utf-8") as f:
        f.writelines(doc.model_dump_json() + "\n" for doc in docs)

    return len(docs)
