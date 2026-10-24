import pytest

from ragbench.chunking import DocumentChunker


def test_chunk_boundaries_and_offsets():
    text = (
        "First sentence defines problem. Second sentence provides empirical methodology. "
        "Third sentence details quantitative results. Fourth sentence provides discussion."
    )
    chunker = DocumentChunker(chunk_size=20, overlap=0)
    chunks = chunker.chunk_document(doc_id="test_doc", text=text)

    assert len(chunks) >= 2
    for chunk in chunks:
        # Check character boundary correctness
        extracted = text[chunk.start_char : chunk.end_char]
        assert extracted == chunk.text
        assert chunk.end_char <= len(text)
        assert chunk.start_char >= 0


def test_token_limit_enforcement():
    text = "Word " * 500
    chunker = DocumentChunker(chunk_size=128, overlap=32)
    chunks = chunker.chunk_document(doc_id="long_doc", text=text)

    assert len(chunks) > 1
    for chunk in chunks:
        tokens = chunker.tokenizer.encode(chunk.text)
        assert len(tokens) <= 128


def test_overlap_validation():
    with pytest.raises(ValueError, match="must be 0 <= overlap < chunk_size"):
        DocumentChunker(chunk_size=128, overlap=128)

    with pytest.raises(ValueError, match="must be 0 <= overlap < chunk_size"):
        DocumentChunker(chunk_size=128, overlap=150)

    with pytest.raises(ValueError, match="must be 0 <= overlap < chunk_size"):
        DocumentChunker(chunk_size=128, overlap=-1)


def test_multisentence_overlap_progress():
    sentences = [f"This is sentence number {i} with some descriptive words." for i in range(50)]
    text = " ".join(sentences)
    chunker = DocumentChunker(chunk_size=64, overlap=16)
    chunks = chunker.chunk_document(doc_id="multi_sent", text=text)

    assert len(chunks) > 5
    for i in range(len(chunks) - 1):
        assert chunks[i + 1].start_char > chunks[i].start_char
        assert chunks[i].text == text[chunks[i].start_char : chunks[i].end_char]

