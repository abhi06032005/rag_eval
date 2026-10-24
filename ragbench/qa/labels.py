"""
Dynamic Chunk-Level Evidence Label Derivation.
Derives relevant chunk IDs dynamically for any chunking configuration by
computing interval intersections between chunk character ranges and evidence spans.
"""

from pydantic import BaseModel, Field

from ragbench.chunking import Chunk


class EvidenceSpan(BaseModel):
    doc_id: str
    start_char: int
    end_char: int
    text: str = ""


class QARecord(BaseModel):
    qa_id: str
    split: str = "dev"  # dev or test
    question: str
    ground_truth_answer: str
    answer_aliases: list[str] = Field(default_factory=list)
    gold_doc_ids: list[str] = Field(default_factory=list)
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    question_type: str = "factoid"
    source_domain: str = "wikipedia"
    creation_method: str = "synthetic"  # synthetic or human
    review_status: str = "approved"


def derive_relevant_chunk_ids(
    chunks: list[Chunk],
    evidence_spans: list[EvidenceSpan],
    min_overlap_chars: int = 1,
) -> list[str]:
    """
    Intersect chunk character spans [start_char, end_char] with question evidence spans.
    Returns list of matching chunk_ids.
    """
    relevant_ids: list[str] = []

    for chunk in chunks:
        for span in evidence_spans:
            if chunk.doc_id != span.doc_id:
                continue

            # Compute character overlap interval
            overlap_start = max(chunk.start_char, span.start_char)
            overlap_end = min(chunk.end_char, span.end_char)

            if (overlap_end - overlap_start) >= min_overlap_chars:
                relevant_ids.append(chunk.chunk_id)
                break  # Don't add duplicate chunk_id if it overlaps multiple spans

    return relevant_ids
