import pytest

from ragbench.provenance import CanonicalDocument, validate_licence


def test_core_sources_licence_allowed():
    for src in ["arxiv", "wikipedia", "govinfo"]:
        valid, msg = validate_licence(src, "allowed")
        assert valid is True
        assert "permitted" in msg


def test_un_permission_refusal():
    # Must reject without reference or acknowledgment
    valid, msg = validate_licence(
        "un", "requires_request", permission_reference=None, acknowledge_permission=False
    )
    assert valid is False
    assert "ERROR: UN full-text reuse is disabled" in msg

    # Valid with explicit reference
    valid, msg = validate_licence(
        "un", "allowed", permission_reference="UNDL-AUTH-2026", acknowledge_permission=True
    )
    assert valid is True
    assert "UNDL-AUTH-2026" in msg


def test_empty_document_rejection():
    with pytest.raises(ValueError, match="Document text must not be empty"):
        CanonicalDocument(
            doc_id="test:1",
            source="arxiv",
            source_record_id="1",
            title="Empty",
            text="   ",
            language="en",
            published_at="2026-01-01",
            retrieved_at="2026-01-01",
            source_locator="http://example.com",
            licence="test",
            snapshot_id="s1",
            content_sha256="h1",
        )
