from ragbench.evaluation.ann import calculate_ann_recall
from ragbench.evaluation.retrieval import (
    calculate_hit_at_k,
    calculate_mrr,
    calculate_recall_at_k,
)


def test_recall_at_k_known_examples():
    gold = {"c1", "c2"}
    retrieved = ["c1", "c3", "c4", "c2", "c5"]

    assert calculate_recall_at_k(retrieved, gold, k=1) == 0.5
    assert calculate_recall_at_k(retrieved, gold, k=3) == 0.5
    assert calculate_recall_at_k(retrieved, gold, k=4) == 1.0
    assert calculate_recall_at_k(retrieved, gold, k=10) == 1.0


def test_hit_at_k_known_examples():
    gold = {"target"}
    assert calculate_hit_at_k(["other1", "target"], gold, k=1) == 0.0
    assert calculate_hit_at_k(["other1", "target"], gold, k=2) == 1.0


def test_mrr_known_examples():
    gold = {"gold_chunk"}
    # Rank 1 -> MRR = 1.0
    assert calculate_mrr(["gold_chunk", "c2", "c3"], gold) == 1.0
    # Rank 2 -> MRR = 0.5
    assert calculate_mrr(["c1", "gold_chunk", "c3"], gold) == 0.5
    # Rank 4 -> MRR = 0.25
    assert calculate_mrr(["c1", "c2", "c3", "gold_chunk"], gold) == 0.25
    # Not found -> MRR = 0.0
    assert calculate_mrr(["c1", "c2", "c3"], gold) == 0.0


def test_ann_recall_known_examples():
    exact = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"]
    # Perfect match: 10/10
    approx_perfect = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"]
    assert calculate_ann_recall(approx_perfect, exact, k=10) == 1.0

    # 8 out of 10 match: 0.8
    approx_8 = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "x9", "x10"]
    assert calculate_ann_recall(approx_8, exact, k=10) == 0.8

    # Completely disjoint: 0.0
    approx_disjoint = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"]
    assert calculate_ann_recall(approx_disjoint, exact, k=10) == 0.0
