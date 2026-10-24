"""
RAGScaleBench ID Generation and Hashing Utilities.
Provides deterministic, collision-resistant identifiers for documents, chunks, QA items, and runs.
"""

import hashlib
import json
from typing import Any


def make_doc_id(source: str, source_record_id: str) -> str:
    """Format a canonical document ID."""
    clean_source = source.strip().lower()
    clean_id = source_record_id.strip()
    return f"{clean_source}:{clean_id}"


def make_chunk_id(doc_id: str, chunk_size: int, overlap: int, chunk_index: int) -> str:
    """
    Compute a stable, deterministic chunk ID for a given document and chunking config.
    """
    key = f"{doc_id}|sz={chunk_size}|ov={overlap}|idx={chunk_index}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"chunk_{digest}"


def make_qa_id(index: int) -> str:
    """Format a canonical QA identifier."""
    return f"qa-{index:06d}"


def compute_content_sha256(text: str) -> str:
    """Compute SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_dict_sha256(data: dict[str, Any]) -> str:
    """Compute canonical SHA-256 of a JSON-serializable dictionary."""
    canonical_json = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def compute_run_id(
    git_sha: str,
    data_manifest_sha: str,
    config_sha: str,
    qa_sha: str,
) -> str:
    """
    Derive a deterministic run ID from component checksums.
    """
    composite = f"{git_sha}:{data_manifest_sha}:{config_sha}:{qa_sha}"
    digest = hashlib.sha256(composite.encode("utf-8")).hexdigest()[:16]
    return f"run_{digest}"
