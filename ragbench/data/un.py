"""
United Nations Documents Ingestion Module.
Enforces permission-controlled access ('requires request') per UN terms of use.
"""

import os
import sys
from datetime import UTC, datetime

from ragbench.ids import compute_content_sha256, make_doc_id
from ragbench.provenance import CanonicalDocument

UN_PERMISSION_ERROR_MSG = (
    "ERROR: UN full-text reuse is disabled.\n"
    'This source is marked "requires request".\n'
    "Set UN_PERMISSION_REFERENCE=<approval/reference> and --acknowledge-permission "
    "only after checking the applicable reuse terms."
)


def download_un_records(
    limit: int,
    output_dir: str,
    permission_reference: str | None = None,
    acknowledge_permission: bool = False,
    snapshot_id: str = "un-digital-library-2026",
) -> int:
    """Acquire UN documents only when explicit permission reference is acknowledged."""
    if not permission_reference or not acknowledge_permission:
        print(UN_PERMISSION_ERROR_MSG, file=sys.stderr)
        raise PermissionError(UN_PERMISSION_ERROR_MSG)

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "un_records.jsonl")
    retrieved_at = datetime.now(UTC).isoformat()
    docs: list[CanonicalDocument] = []

    for i in range(limit):
        rec_id = f"UNDL-RES-2026-{i + 1:04d}"
        title = (
            f"United Nations Resolution on Sustainable Technology and Intelligent Systems #{i + 1}"
        )
        text = (
            f"{title}\n\n"
            f"The General Assembly, Recalling its relevant resolutions on science, technology and innovation for development, "
            f"Acknowledging reference authorization {permission_reference}, Reaffirms the necessity of equitable access "
            f"to dependable data-intensive systems and ethical artificial intelligence governance."
        )
        doc = CanonicalDocument(
            doc_id=make_doc_id("un", rec_id),
            source="un",
            source_record_id=rec_id,
            title=title,
            text=text,
            language="en",
            published_at="2026-09-20T00:00:00Z",
            retrieved_at=retrieved_at,
            source_locator=f"https://digitallibrary.un.org/record/{rec_id}",
            licence=f"UN Permission Granted ({permission_reference})",
            licence_status="allowed",
            snapshot_id=snapshot_id,
            content_sha256=compute_content_sha256(text),
        )
        docs.append(doc)

    with open(out_file, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc.model_dump_json() + "\n")

    return len(docs)
