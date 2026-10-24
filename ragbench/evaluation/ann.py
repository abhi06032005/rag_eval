"""
Approximate Nearest Neighbour (ANN) Recall.
Compares Qdrant HNSW approximate neighbours against exact brute-force search
to isolate graph/quantisation approximation error from semantic embedding error.
"""

from collections.abc import Sequence


def calculate_ann_recall(
    approx_hit_ids: Sequence[str], exact_hit_ids: Sequence[str], k: int
) -> float:
    """
    ANNRecall@k = |approx_ids[:k] ∩ exact_ids[:k]| / k.
    Measures how many of the exact nearest neighbours were successfully discovered
    by the approximate HNSW / quantised search.
    """
    if k <= 0:
        return 0.0
    top_approx = set(approx_hit_ids[:k])
    top_exact = set(exact_hit_ids[:k])
    overlap = len(top_approx.intersection(top_exact))
    return float(overlap) / float(k)
