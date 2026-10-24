"""
Pareto Analysis and Performance Reporting.
Extracts non-dominated configurations balancing Recall@10, p95 Latency, and Memory.
"""

import pandas as pd


def calculate_pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find non-dominated configurations where:
    Objective 1: Maximize gold_recall_at_10
    Objective 2: Minimize retrieval_p95_ms
    Objective 3: Minimize qdrant_peak_rss_mb
    """
    if df.empty:
        return df

    candidates = df.copy()
    is_pareto = []

    for i, row in candidates.iterrows():
        dominated = False
        r_rec = row.get("gold_recall_at_10", 0.0)
        r_lat = row.get("retrieval_p95_ms", float("inf"))
        r_mem = row.get("qdrant_peak_rss_mb", float("inf"))

        for j, other in candidates.iterrows():
            if i == j:
                continue
            o_rec = other.get("gold_recall_at_10", 0.0)
            o_lat = other.get("retrieval_p95_ms", float("inf"))
            o_mem = other.get("qdrant_peak_rss_mb", float("inf"))

            # Other dominates row if it is >= recall, <= latency, <= memory, and strictly better in at least one
            if (o_rec >= r_rec and o_lat <= r_lat and o_mem <= r_mem) and (
                o_rec > r_rec or o_lat < r_lat or o_mem < r_mem
            ):
                dominated = True
                break

        is_pareto.append(not dominated)

    candidates["is_pareto"] = is_pareto
    return candidates[candidates["is_pareto"]].copy()
