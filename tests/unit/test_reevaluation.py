import pytest

from ragbench.evaluation.bm25 import BM25Okapi
from ragbench.retrieval.policies import AdaptiveRouterPolicy, AdaptiveThresholds, FixedHybridPolicy
from scripts.full_beir_reevaluation import metrics


def test_graded_ndcg_and_precision_denominator():
    scores = metrics(["low", "high"], {"high": 2, "low": 1})
    assert scores["nDCG@10"] < scores["binary_nDCG@10"] == 1
    assert scores["Precision@10"] == 0.2
    assert scores["Recall@10"] == 1


def test_duplicate_ranking_rejected():
    with pytest.raises(AssertionError):
        metrics(["a", "a"], {"a": 1})


@pytest.mark.parametrize("kind", [FixedHybridPolicy, AdaptiveRouterPolicy])
def test_explicit_candidate_depth_independent_of_return_cutoff(kind):
    engine = BM25Okapi()
    engine.fit([(str(i), "shared query") for i in range(300)])
    calls = []

    def dense(v, k):
        calls.append(k)
        return [(str(i), 1 / (i + 1)) for i in reversed(range(k))]

    policy = kind(engine, dense, candidate_depth=250)
    first = policy.retrieve("shared query", [1.0], top_k=10)
    second = policy.retrieve("shared query", [1.0], top_k=50)
    assert calls == [250, 250]
    assert first.retrieved_docs == second.retrieved_docs[:10]
    with pytest.raises(ValueError):
        policy.retrieve("shared query", [1.0], top_k=251)
