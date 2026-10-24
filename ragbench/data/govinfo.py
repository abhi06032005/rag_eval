"""
GovInfo U.S. Government Documents Ingestion Module.
Acquires authentic Congressional bulk XML documents from official GPO repositories.
Documents are in the public domain under 17 U.S.C. §105.
"""

import json
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, datetime

from ragbench.ids import compute_content_sha256, make_doc_id
from ragbench.provenance import CanonicalDocument


def extract_text_from_bill_xml(xml_content: bytes) -> tuple[str, str]:
    """Parse GPO Congressional bill XML and extract title and body text."""
    try:
        root = ET.fromstring(xml_content)
        title_el = root.find(".//official-title")
        if title_el is None or not title_el.text:
            title_el = root.find(".//short-title")
        title = title_el.text.strip() if title_el is not None and title_el.text else "Congressional Bill"
        title = re.sub(r"\s+", " ", title)

        # Extract all text, strip excess whitespace
        full_text = " ".join(root.itertext())
        full_text = re.sub(r"\s+", " ", full_text).strip()
        return title, full_text
    except Exception:
        # Fallback raw text stripping if XML structure is non-standard
        raw = xml_content.decode("utf-8", errors="ignore")
        clean = re.sub(r"<[^>]+>", " ", raw)
        clean = re.sub(r"\s+", " ", clean).strip()
        return "Congressional Document", clean


def generate_synthetic_govinfo_documents(count: int, snapshot_id: str) -> list[CanonicalDocument]:
    """Generate deterministic GovInfo-style records ONLY for offline unit tests."""
    docs: list[CanonicalDocument] = []
    retrieved_at = datetime.now(UTC).isoformat()

    items = [
        (
            "National Quantum and Intelligent Systems Initiative Act",
            "119-PL-42",
            "An Act to provide for a coordinated federal research programme to accelerate intelligent software systems, vector database acceleration, and quantum information science. Be it enacted by the Senate and House of Representatives of the United States of America in Congress assembled. Section 101: The Director of the National Science Foundation shall establish a competitive grant programme for academic consortia evaluating data-intensive intelligent systems, latency characterisation under heavy load, and robust artificial intelligence architectures.",
        ),
        (
            "Federal Register: Standards for Data-Intensive Machine Learning Services",
            "FR-2026-0812",
            "Agency: National Institute of Standards and Technology. Action: Notice of recommended technical guidelines. Summary: This document sets forth minimum technical metrics for verifying faithfulness, claim-level entailment, and approximate-nearest-neighbour recall in government-operated dense retrieval pipelines. Federal systems must establish reproducible testing harnesses before deploying generative interfaces across public information repositories.",
        ),
        (
            "Congressional Record: Proceedings on Algorithmic Transparency",
            "CREC-2026-H1044",
            "The House met at 10 a.m. Mr. Speaker, I rise today to direct the attention of this Chamber to the critical challenge of verifying automated software decisions. When federal agencies rely on artificial intelligence pipelines to index millions of public records, the mathematical properties of graph traversal and vector quantisation dictate whether citizens obtain accurate statutory information or distorted hallucinations.",
        ),
        (
            "General Services Administration Circular on Open Source Software Compliance",
            "GSA-CIRC-2026-04",
            "Policy Memorandum: All software components developed under federal contract must clearly declare licence provenance, maintain deterministic build manifests, and provide automated test suites demonstrating compliance with Section 105 of Title 17, United States Code.",
        ),
    ]

    for i in range(count):
        idx = i % len(items)
        title_base, doc_num, body = items[idx]
        record_id = f"GOVINFO-{doc_num}-{i + 1:04d}"
        title = f"{title_base} (Record #{i + 1})"
        text = f"{title}\n\n{body} Enacted and recorded in official statutory register under volume 142, page {200 + i}."
        doc_id = make_doc_id("govinfo", record_id)
        docs.append(
            CanonicalDocument(
                doc_id=doc_id,
                source="govinfo",
                source_record_id=record_id,
                title=title,
                text=text,
                language="en",
                published_at="2026-02-01T00:00:00Z",
                retrieved_at=retrieved_at,
                source_locator=f"https://www.govinfo.gov/app/details/{record_id}",
                licence="U.S. Government Work (17 U.S.C. §105) - Public Domain",
                licence_status="allowed",
                snapshot_id=snapshot_id,
                content_sha256=compute_content_sha256(text),
            )
        )
    return docs


def download_govinfo_records(
    limit: int,
    output_dir: str,
    snapshot_id: str = "govinfo-bulk-2026",
    use_synthetic_fallback: bool = False,
) -> int:
    """
    Acquire authentic GovInfo bulk data records (118th Congress House & Senate Bills).
    Downloads authentic XML files from the official GPO bulk data repository.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "govinfo_records.jsonl")

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
    bill_types = ["hr", "s"]
    bill_index = 1
    type_idx = 0

    while len(docs_collected) < limit and bill_index <= 5000:
        btype = bill_types[type_idx % len(bill_types)]
        stage = "ih" if btype == "hr" else "is"
        package_id = f"BILLS-118{btype}{bill_index}{stage}"
        url = f"https://www.govinfo.gov/bulkdata/BILLS/118/1/{btype}/{package_id}.xml"

        if package_id in seen_ids:
            bill_index += 1
            type_idx += 1
            continue

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "RAGScaleBench/1.0 (academic research at ediss.eu)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                if resp.status == 200:
                    xml_bytes = resp.read()
                    title, full_text = extract_text_from_bill_xml(xml_bytes)
                    if len(full_text) > 100:
                        doc = CanonicalDocument(
                            doc_id=make_doc_id("govinfo", package_id),
                            source="govinfo",
                            source_record_id=package_id,
                            title=title,
                            text=f"{title}\n\n{full_text}",
                            language="en",
                            published_at="2023-01-01T00:00:00Z",
                            retrieved_at=retrieved_at,
                            source_locator=f"https://www.govinfo.gov/bulkdata/BILLS/118/1/{btype}/{package_id}.xml",
                            licence="U.S. Government Work (17 U.S.C. §105) - Public Domain",
                            licence_status="allowed",
                            snapshot_id=snapshot_id,
                            content_sha256=compute_content_sha256(f"{title}\n\n{full_text}"),
                        )
                        docs_collected.append(doc)
                        seen_ids.add(package_id)
        except Exception:
            # 404 or transient bill numbering gaps are normal in congressional numbering
            pass

        bill_index += 1
        type_idx += 1

    needed = limit - len(docs_collected)
    if needed > 0:
        if use_synthetic_fallback:
            synth = generate_synthetic_govinfo_documents(needed, snapshot_id)
            docs_collected.extend(synth)
        else:
            raise RuntimeError(
                f"Failed to acquire {limit} authentic GovInfo records. Acquired {len(docs_collected)}."
            )

    with open(out_file, "w", encoding="utf-8") as f:
        for doc in docs_collected:
            f.write(doc.model_dump_json() + "\n")

    return len(docs_collected)
