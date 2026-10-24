"""
English Wikipedia Corpus Ingestion Module.
Preserves CC BY-SA 4.0 / GFDL provenance and attribution locators.
Acquires authentic live MediaWiki articles via paginated REST API.
"""

import json
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime

from ragbench.ids import compute_content_sha256, make_doc_id
from ragbench.provenance import CanonicalDocument


def clean_wikitext_or_html(text: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", clean)
    clean = re.sub(r"\{\{[^\}]+\}\}", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def generate_synthetic_wikipedia_documents(count: int, snapshot_id: str) -> list[CanonicalDocument]:
    """Generate deterministic general knowledge Wikipedia-style articles ONLY for tests."""
    docs: list[CanonicalDocument] = []
    retrieved_at = datetime.now(UTC).isoformat()

    topics = [
        (
            "Distributed Computing Architecture",
            "Distributed computing is a field of computer science that studies distributed systems. The components of a distributed system communicate and coordinate their actions by passing messages. The components interact with each other in order to achieve a common goal. Three significant challenges of distributed systems are: maintaining concurrency of components, overcoming lack of a global clock, and handling independent failure of components. Modern vector databases apply distributed architectures to partition high-dimensional vector indices across multiple storage shards.",
        ),
        (
            "Vector Quantisation in Signal Processing and Machine Learning",
            "Vector quantisation is a classical quantisation technique from signal processing that allows the modelling of probability density functions by the distribution of prototype vectors. It works by dividing a large set of points into groups having approximately the same number of points closest to them. In contemporary vector search systems, product quantisation decomposes a high-dimensional vector into lower-dimensional sub-vectors and quantises each subspace independently, substantially lowering RAM consumption at the cost of slight recall reduction.",
        ),
        (
            "Information Retrieval and Evaluation Metrics",
            "Information retrieval is the science of searching for information in a document, searching for documents themselves, and also searching for the metadata that describes data. Common evaluation metrics include Precision at k, Recall at k, Mean Reciprocal Rank (MRR), and Normalised Discounted Cumulative Gain (nDCG). In dense retrieval systems, lexical metrics are frequently complemented by semantic similarity checks.",
        ),
        (
            "Open Source Software Licences and Attribution",
            "Open-source licences are a type of licence for computer software and other products that allows the source code, blueprint or design to be used, modified and shared under defined terms and conditions. The Creative Commons Attribution-ShareAlike (CC BY-SA) licence allows re-users to distribute, remix, adapt, and build upon the material in any medium or format, so long as attribution is given to the creator.",
        ),
        (
            "Erasmus Mundus Joint Master Degrees",
            "Erasmus Mundus Joint Masters are prestigious international study programmes, jointly designed and delivered by an international consortium of higher education institutions. The European Master in Dependable Intelligent Software Systems (EDISS) educates specialists in data-intensive intelligent systems, combining software engineering with machine learning and systems research.",
        ),
    ]

    for i in range(count):
        topic_idx = i % len(topics)
        title_base, body = topics[topic_idx]
        page_id = f"enwiki_{i + 50000}"
        title = f"{title_base} (Topic Section {i + 1})"
        text = f"{title}\n\n{body} Detailed empirical documentation and cross-references for entry {i + 1}."
        doc_id = make_doc_id("wikipedia", page_id)
        docs.append(
            CanonicalDocument(
                doc_id=doc_id,
                source="wikipedia",
                source_record_id=page_id,
                title=title,
                text=text,
                language="en",
                published_at="2026-10-01T00:00:00Z",
                retrieved_at=retrieved_at,
                source_locator=f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title_base)}",
                licence="CC BY-SA 4.0 / GFDL with attribution",
                licence_status="allowed",
                snapshot_id=snapshot_id,
                content_sha256=compute_content_sha256(text),
            )
        )
    return docs


def download_wikipedia_records(
    limit: int,
    output_dir: str,
    snapshot_id: str = "enwiki-20261001",
    use_synthetic_fallback: bool = False,
) -> int:
    """Acquire authentic English Wikipedia articles adhering to CC BY-SA terms."""
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "wikipedia_records.jsonl")

    # Resumable loading
    docs_collected: list[CanonicalDocument] = []
    seen_ids: set[str] = set()
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    doc = CanonicalDocument.model_validate_json(line)
                    docs_collected.append(doc)
                    seen_ids.add(doc.source_record_id)

    if len(docs_collected) >= limit:
        return len(docs_collected)

    retrieved_at = datetime.now(UTC).isoformat()
    max_retries = 20

    while len(docs_collected) < limit and max_retries > 0:
        batch_needed = min(limit - len(docs_collected), 50)
        api_url = (
            "https://en.wikipedia.org/w/api.php?action=query&generator=random&grnnamespace=0"
            f"&grnlimit={batch_needed}&prop=extracts|info&explaintext=1&exintro=1&format=json"
        )
        req = urllib.request.Request(
            api_url, headers={"User-Agent": "RAGScaleBench/1.0 (academic research at ediss.eu)"}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for pid, page in pages.items():
                if str(pid) in seen_ids:
                    continue
                title = page.get("title", "")
                extract = page.get("extract", "").strip()
                if not extract or len(extract) < 50:
                    continue
                full_text = f"{title}\n\n{extract}"
                doc = CanonicalDocument(
                    doc_id=make_doc_id("wikipedia", str(pid)),
                    source="wikipedia",
                    source_record_id=str(pid),
                    title=title,
                    text=full_text,
                    language="en",
                    published_at="2026-10-01T00:00:00Z",
                    retrieved_at=retrieved_at,
                    source_locator=f"https://en.wikipedia.org/?curid={pid}",
                    licence="CC BY-SA 4.0 / GFDL with attribution",
                    licence_status="allowed",
                    snapshot_id=snapshot_id,
                    content_sha256=compute_content_sha256(full_text),
                )
                docs_collected.append(doc)
                seen_ids.add(str(pid))
                if len(docs_collected) >= limit:
                    break
            time.sleep(0.2)  # courteous MediaWiki rate-limiting
        except Exception:
            max_retries -= 1
            time.sleep(0.5)

    needed = limit - len(docs_collected)
    if needed > 0:
        if use_synthetic_fallback:
            synth = generate_synthetic_wikipedia_documents(needed, snapshot_id)
            docs_collected.extend(synth)
        else:
            raise RuntimeError(
                f"Failed to acquire {limit} authentic Wikipedia articles. Acquired {len(docs_collected)}."
            )

    with open(out_file, "w", encoding="utf-8") as f:
        for doc in docs_collected:
            f.write(doc.model_dump_json() + "\n")

    return len(docs_collected)
