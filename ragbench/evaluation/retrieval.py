"""
Labelled Retrieval Evaluation Metrics.
Calculates Recall@k, Hit@k, MRR, and nDCG@k against dynamic chunk-level gold labels.
"""

import math
from collections.abc import Sequence


def calculate_recall_at_k(retrieved_ids: Sequence[str], gold_ids: set[str], k: int) -> float:
    """Recall@k = |gold_ids ∩ retrieved_ids[:k]| / |gold_ids|."""
    if not gold_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    matched = sum(1 for cid in top_k if cid in gold_ids)
    return float(matched) / float(len(gold_ids))


def calculate_hit_at_k(retrieved_ids: Sequence[str], gold_ids: set[str], k: int) -> float:
    """Hit@k = 1.0 if any gold item in retrieved[:k], else 0.0."""
    if not gold_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return 1.0 if any(cid in gold_ids for cid in top_k) else 0.0


def calculate_mrr(retrieved_ids: Sequence[str], gold_ids: set[str]) -> float:
    """Mean Reciprocal Rank (MRR) of first relevant retrieved chunk."""
    if not gold_ids:
        return 0.0
    for rank, cid in enumerate(retrieved_ids, 1):
        if cid in gold_ids:
            return 1.0 / rank
    return 0.0


def calculate_ndcg_at_k(retrieved_ids: Sequence[str], gold_ids: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank k."""
    if not gold_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    dcg = 0.0
    for rank, cid in enumerate(top_k, 1):
        if cid in gold_ids:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(gold_ids), k) + 1))
    return (dcg / idcg) if idcg > 0.0 else 0.0
