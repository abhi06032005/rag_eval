from ragbench.chunking import Chunk
from ragbench.qa.labels import EvidenceSpan, derive_relevant_chunk_ids


def test_dynamic_evidence_span_to_chunk_mapping():
    chunks = [
        Chunk(
            chunk_id="chunk_0",
            doc_id="doc_A",
            chunk_index=0,
            start_char=0,
            end_char=100,
            start_token=0,
            end_token=25,
            text="...",
        ),
        Chunk(
            chunk_id="chunk_1",
            doc_id="doc_A",
            chunk_index=1,
            start_char=90,
            end_char=200,
            start_token=22,
            end_token=50,
            text="...",
        ),
        Chunk(
            chunk_id="chunk_2",
            doc_id="doc_B",
            chunk_index=0,
            start_char=0,
            end_char=100,
            start_token=0,
            end_token=25,
            text="...",
        ),
    ]

    # Span in doc_A at [120, 150] should only match chunk_1
    spans_1 = [EvidenceSpan(doc_id="doc_A", start_char=120, end_char=150)]
    rel_1 = derive_relevant_chunk_ids(chunks, spans_1)
    assert rel_1 == ["chunk_1"]

    # Span in doc_A at [95, 105] overlaps both chunk_0 and chunk_1 due to chunk overlap!
    spans_2 = [EvidenceSpan(doc_id="doc_A", start_char=95, end_char=105)]
    rel_2 = derive_relevant_chunk_ids(chunks, spans_2)
    assert set(rel_2) == {"chunk_0", "chunk_1"}

    # Span in doc_B should match chunk_2
    spans_3 = [EvidenceSpan(doc_id="doc_B", start_char=10, end_char=50)]
    rel_3 = derive_relevant_chunk_ids(chunks, spans_3)
    assert rel_3 == ["chunk_2"]
