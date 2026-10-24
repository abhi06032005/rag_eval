"""
RAGScaleBench Provenance and Licence Validation Module.
Enforces legal provenance, schema validation, and manifest generation.
"""

import os
from datetime import UTC, datetime
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

LicenceStatus = Literal["allowed", "requires_request", "restricted", "allowed_conditional"]


class CanonicalDocument(BaseModel):
    """
    Canonical schema for all acquired documents across all corpus sources.
    """

    doc_id: str
    source: str
    source_record_id: str
    title: str
    text: str
    language: str = "en"
    published_at: str
    retrieved_at: str
    source_locator: str
    licence: str
    licence_status: LicenceStatus = "allowed"
    snapshot_id: str
    content_sha256: str

    @field_validator("text")
    @classmethod
    def validate_non_empty_text(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Document text must not be empty")
        return cleaned

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        if v.lower() != "en":
            raise ValueError(f"Only English documents are accepted in core benchmark, got: {v}")
        return "en"


class CorpusManifest(BaseModel):
    """
    Corpus lock manifest containing exact snapshot IDs, document counts, and checksums.
    """

    manifest_version: int = 1
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    total_documents: int
    corpus_sha256: str
    source_counts: dict[str, int]
    source_snapshots: dict[str, str]
    licence_declarations: dict[str, str]


def validate_licence(
    source: str,
    licence_status: str,
    permission_reference: str | None = None,
    acknowledge_permission: bool = False,
) -> tuple[bool, str]:
    """
    Enforce licence compliance for corpus ingestion.
    """
    source = source.lower()
    if source in ("arxiv", "wikipedia", "govinfo"):
        return True, f"{source} is permitted under standard benchmark terms."

    if source == "commoncrawl":
        return True, "Common Crawl WET allowed via domain allowlist with attribution."

    if source == "un":
        if not permission_reference or not acknowledge_permission:
            error_msg = (
                "ERROR: UN full-text reuse is disabled.\n"
                'This source is marked "requires request".\n'
                "Set UN_PERMISSION_REFERENCE=<approval/reference> and --acknowledge-permission "
                "only after checking the applicable reuse terms."
            )
            return False, error_msg
        return True, f"UN full-text ingestion approved with reference: {permission_reference}"

    return False, f"Unknown source '{source}' has unvalidated licence terms."


def write_manifest_lock(manifest: CorpusManifest, path: str) -> None:
    """Write locked manifest to YAML."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest.model_dump(), f, sort_keys=False)


def load_manifest_lock(path: str) -> CorpusManifest:
    """Load and parse locked manifest."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return CorpusManifest(**data)
