"""
Pytest Fixtures for RAGScaleBench.
"""

import pytest

from ragbench.provenance import CanonicalDocument
from ragbench.qa.labels import EvidenceSpan, QARecord


@pytest.fixture
def sample_canonical_doc():
    text = (
        "Vector index quantisation enables high-density search. "
        "Product quantisation partitions high-dimensional vectors into orthogonal subspaces. "
        "Scalar quantisation casts 32-bit floats into 8-bit integers."
    )
    return CanonicalDocument(
        doc_id="arxiv:2601.99999",
        source="arxiv",
        source_record_id="2601.99999",
        title="Sample Quantisation Study",
        text=text,
        language="en",
        published_at="2026-01-01T00:00:00Z",
        retrieved_at="2026-10-21T00:00:00Z",
        source_locator="https://arxiv.org/abs/2601.99999",
        licence="arXiv Non-exclusive license to distribute",
        licence_status="allowed",
        snapshot_id="arxiv-test",
        content_sha256="fakehash123",
    )


@pytest.fixture
def sample_qa_record(sample_canonical_doc):
    fact = "Product quantisation partitions high-dimensional vectors into orthogonal subspaces."
    start = sample_canonical_doc.text.find(fact)
    end = start + len(fact)
    return QARecord(
        qa_id="qa-000001",
        split="test",
        question="What does product quantisation do?",
        ground_truth_answer=fact,
        answer_aliases=[fact.lower()],
        gold_doc_ids=[sample_canonical_doc.doc_id],
        evidence_spans=[
            EvidenceSpan(
                doc_id=sample_canonical_doc.doc_id,
                start_char=start,
                end_char=end,
                text=fact,
            )
        ],
        question_type="factoid",
        source_domain="arxiv",
        creation_method="human",
        review_status="approved",
    )
