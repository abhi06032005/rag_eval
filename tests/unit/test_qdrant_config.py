from qdrant_client.http import models as rest

from ragbench.vectorstores.qdrant import QdrantVectorStore


def test_qdrant_quantization_translation():
    qs = QdrantVectorStore(in_memory=True)

    # None
    assert qs._map_quantization("none", rescore=True) is None

    # Scalar
    sq = qs._map_quantization("scalar", rescore=True)
    assert isinstance(sq, rest.ScalarQuantization)
    assert sq.scalar.type == rest.ScalarType.INT8

    # Binary
    bq = qs._map_quantization("binary", rescore=True)
    assert isinstance(bq, rest.BinaryQuantization)

    # Product Quantisation
    pq = qs._map_quantization("product_x16", rescore=True)
    assert isinstance(pq, rest.ProductQuantization)
    assert pq.product.compression == rest.CompressionRatio.X16


def test_qdrant_distance_mapping():
    qs = QdrantVectorStore(in_memory=True)
    assert qs._map_distance("cosine") == rest.Distance.COSINE
    assert qs._map_distance("dot") == rest.Distance.DOT
    assert qs._map_distance("euclidean") == rest.Distance.EUCLID
