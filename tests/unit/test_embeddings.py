import numpy as np
import pytest

from ragbench.embeddings.sentence_transformers import OfflineHashedEmbedder, l2_normalize


def test_embedding_dimensionality():
    emb384 = OfflineHashedEmbedder("test-384", dimension=384)
    v1 = emb384.embed_query("dense retrieval")
    assert len(v1) == 384

    emb1024 = OfflineHashedEmbedder("BAAI/bge-m3")
    v2 = emb1024.embed_query("multilingual retrieval")
    assert len(v2) == 1024


def test_embedding_normalisation():
    raw = np.array([3.0, 4.0], dtype=np.float32)
    normed = l2_normalize(raw)
    assert pytest.approx(np.linalg.norm(normed), rel=1e-5) == 1.0

    emb = OfflineHashedEmbedder("test-384", dimension=384)
    v = emb.embed_query("sample text for unit norm check")
    norm = sum(x**2 for x in v)
    assert pytest.approx(norm, rel=1e-4) == 1.0
