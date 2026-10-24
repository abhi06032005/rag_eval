"""
Unit tests for retrieval policies and adaptive router.
"""

import numpy as np
import pytest
from ragbench.evaluation.bm25 import BM25Okapi
from ragbench.retrieval.policies import (
    AdaptiveRouterPolicy,
    AdaptiveThresholds,
    BM25Policy,
    DensePolicy,
    FixedHybridPolicy,
)


@pytest.fixture
def sample_corpus():
    return [
        ("doc1", "CRISPR-Cas9 gene editing enables targeted modifications in the genome."),
        ("doc2", "Deep learning transformer models revolutionize natural language processing."),
        ("doc3", "Bacterial resistance to penicillin is mediated by beta-lactamase enzymes."),
        ("doc4", "Fiscal policy and interest rate changes influence corporate bond yields."),
    ]


@pytest.fixture
def bm25_engine(sample_corpus):
    engine = BM25Okapi()
    engine.fit(sample_corpus)
    return engine


def dummy_dense_search(query_vec: np.ndarray, top_k: int):
    # Dummy dense search returning deterministic rankings
    return [("doc2", 0.88), ("doc1", 0.45), ("doc3", 0.20)][:top_k]


def test_bm25_policy(bm25_engine):
    policy = BM25Policy(bm25_engine)
    res = policy.retrieve("penicillin beta-lactamase", top_k=2)
    assert res.route_taken == "bm25"
    assert res.bm25_search_executed is True
    assert res.dense_search_executed is False
    assert res.retrieved_docs[0][0] == "doc3"


def test_dense_policy():
    policy = DensePolicy(dummy_dense_search)
    dummy_vec = np.ones(384)
    res = policy.retrieve("anything", query_vec=dummy_vec, top_k=2)
    assert res.route_taken == "dense"
    assert res.bm25_search_executed is False
    assert res.dense_search_executed is True
    assert res.retrieved_docs[0][0] == "doc2"


def test_fixed_hybrid_policy(bm25_engine):
    policy = FixedHybridPolicy(bm25_engine, dummy_dense_search, k_rrf=60)
    dummy_vec = np.ones(384)
    res = policy.retrieve("transformer deep learning", query_vec=dummy_vec, top_k=2)
    assert res.route_taken == "hybrid"
    assert res.bm25_search_executed is True
    assert res.dense_search_executed is True
    assert len(res.retrieved_docs) > 0


def test_adaptive_router_lexical_route(bm25_engine):
    router = AdaptiveRouterPolicy(
        bm25_engine,
        dummy_dense_search,
        thresholds=AdaptiveThresholds(tau_lexical_margin=0.20, tau_min_idf=1.0, tau_bm25_weak=1.0),
    )
    # Query with exact rare words that cleanly dominate
    res = router.retrieve("penicillin beta-lactamase enzymes", top_k=2)
    assert res.route_taken == "bm25"
    assert res.dense_search_executed is False
    assert "high_lexical_margin" in res.reason


def test_adaptive_router_dense_dominance(bm25_engine):
    router = AdaptiveRouterPolicy(
        bm25_engine,
        dummy_dense_search,
        thresholds=AdaptiveThresholds(tau_lexical_margin=0.99, tau_min_idf=100.0, tau_dense_confident=0.75),
    )
    dummy_vec = np.ones(384)
    res = router.retrieve("modifications and changes", query_vec=dummy_vec, top_k=2)
    # Dense top score is 0.88 >= 0.75 while BM25 is weak
    assert res.route_taken == "dense"
    assert res.dense_search_executed is True
    assert "dense_semantic_dominance" in res.reason


def test_adaptive_router_hybrid_fallback(bm25_engine):
    # If dense similarity is moderate (e.g., 0.65 < tau_dense_confident=0.75)
    def moderate_dense_search(query_vec: np.ndarray, top_k: int):
        return [("doc2", 0.65), ("doc1", 0.45)][:top_k]

    router = AdaptiveRouterPolicy(
        bm25_engine,
        moderate_dense_search,
        thresholds=AdaptiveThresholds(tau_lexical_margin=0.99, tau_min_idf=100.0, tau_dense_confident=0.75),
    )
    dummy_vec = np.ones(384)
    res = router.retrieve("modifications and changes", query_vec=dummy_vec, top_k=2)
    assert res.route_taken == "hybrid"
    assert res.dense_search_executed is True
    assert "uncertainty_fallback_hybrid" in res.reason
