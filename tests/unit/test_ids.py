from ragbench.ids import (
    compute_content_sha256,
    compute_run_id,
    make_chunk_id,
    make_doc_id,
    make_qa_id,
)


def test_stable_doc_ids():
    id1 = make_doc_id("ARXIV ", "2601.01234")
    id2 = make_doc_id("arxiv", "2601.01234")
    assert id1 == "arxiv:2601.01234"
    assert id1 == id2


def test_stable_chunk_ids():
    c1 = make_chunk_id("doc_1", chunk_size=256, overlap=32, chunk_index=0)
    c2 = make_chunk_id("doc_1", chunk_size=256, overlap=32, chunk_index=0)
    c3 = make_chunk_id("doc_1", chunk_size=256, overlap=32, chunk_index=1)
    c4 = make_chunk_id("doc_1", chunk_size=128, overlap=32, chunk_index=0)

    assert c1 == c2
    assert c1 != c3
    assert c1 != c4
    assert c1.startswith("chunk_")


def test_qa_id_formatting():
    assert make_qa_id(1) == "qa-000001"
    assert make_qa_id(142) == "qa-000142"


def test_content_sha256():
    text = "Hello world"
    h1 = compute_content_sha256(text)
    h2 = compute_content_sha256(text)
    assert len(h1) == 64
    assert h1 == h2


def test_compute_run_id():
    run_id = compute_run_id("git123", "manifest456", "cfg789", "qa000")
    assert run_id.startswith("run_")
    assert len(run_id) == 20
