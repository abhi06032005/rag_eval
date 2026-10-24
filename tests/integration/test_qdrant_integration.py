from ragbench.vectorstores.base import IndexConfig, SearchConfig, VectorRecord
from ragbench.vectorstores.qdrant import QdrantVectorStore


def test_qdrant_exact_vs_ann_and_threshold():
    store = QdrantVectorStore(in_memory=True)
    idx_cfg = IndexConfig(
        collection_name="integration_test_col",
        dimension=4,
        distance="cosine",
        m=16,
        ef_construct=64,
        quantization="none",
    )
    store.create_index(idx_cfg)

    # 4 orthogonal points
    pts = [
        VectorRecord(id="chunk_a", vector=[1.0, 0.0, 0.0, 0.0], payload={"text": "document A"}),
        VectorRecord(id="chunk_b", vector=[0.0, 1.0, 0.0, 0.0], payload={"text": "document B"}),
        VectorRecord(id="chunk_c", vector=[0.0, 0.0, 1.0, 0.0], payload={"text": "document C"}),
        VectorRecord(id="chunk_d", vector=[0.0, 0.0, 0.0, 1.0], payload={"text": "document D"}),
    ]
    store.upsert(pts)

    # Query matching chunk_a exactly
    query = [1.0, 0.0, 0.0, 0.0]
    hits = store.search(query, SearchConfig(k=2, hnsw_ef=32))
    assert len(hits) == 2
    assert hits[0].id == "chunk_a"
    assert hits[0].score >= 0.99

    # Exact search reference
    exact_hits = store.exact_search(query, k=2)
    assert len(exact_hits) == 2
    assert exact_hits[0].id == "chunk_a"

    # Threshold filtering (only chunk_a has cosine > 0.5)
    thresh_hits = store.search(query, SearchConfig(k=4, score_threshold=0.5))
    assert len(thresh_hits) == 1
    assert thresh_hits[0].id == "chunk_a"

    store.delete_index()
