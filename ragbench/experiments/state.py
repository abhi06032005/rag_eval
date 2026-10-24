"""
Experiment State Management and Atomic Result Persistence.
Maintains results/summary.csv and per-run queries.parquet.
Ensures crash-resilience and idempotency.
"""

import csv
import os
import tempfile
from typing import Any

import pandas as pd

SUMMARY_HEADER = [
    "run_id",
    "git_sha",
    "config_sha",
    "data_manifest_sha",
    "hardware_id",
    "timestamp",
    "documents",
    "chunks",
    "embedding_model",
    "embedding_dimensions",
    "chunk_size",
    "overlap",
    "hnsw_m",
    "ef_construct",
    "hnsw_ef",
    "quantization",
    "pq_compression",
    "k",
    "score_threshold",
    "ann_recall_at_10",
    "gold_recall_at_1",
    "gold_recall_at_5",
    "gold_recall_at_10",
    "gold_recall_at_20",
    "mrr",
    "ndcg_at_10",
    "exact_match",
    "token_f1",
    "rouge_l",
    "bleu",
    "answer_correctness",
    "context_precision",
    "context_recall",
    "faithfulness",
    "entailment_rate",
    "unsupported_claim_rate",
    "retrieval_p50_ms",
    "retrieval_p95_ms",
    "retrieval_p99_ms",
    "generation_p95_ms",
    "end_to_end_p95_ms",
    "qdrant_peak_rss_mb",
    "api_peak_rss_mb",
    "generator_peak_rss_mb",
    "index_size_mb",
    "index_build_seconds",
    "throughput_rps",
    "failure_rate",
    "dropped_iterations",
]


class ExperimentStateManager:
    """Manages persistent results and resume status."""

    def __init__(self, summary_csv_path: str = "results/summary.csv"):
        self.summary_path = summary_csv_path
        os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
        self._ensure_summary_file()

    def _ensure_summary_file(self) -> None:
        if not os.path.exists(self.summary_path):
            with open(self.summary_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(SUMMARY_HEADER)

    def get_completed_config_hashes(self) -> set[str]:
        """Read all config hashes already present in summary.csv."""
        if not os.path.exists(self.summary_path):
            return set()
        df = pd.read_csv(self.summary_path)
        if "config_sha" in df.columns:
            return set(df["config_sha"].dropna().astype(str).tolist())
        return set()

    def append_summary_row_atomic(self, row_dict: dict[str, Any]) -> None:
        """Atomically append a summary row using a lock/atomic write."""
        row_values = [row_dict.get(col, "") for col in SUMMARY_HEADER]

        # Read existing rows
        existing_rows = []
        if os.path.exists(self.summary_path):
            with open(self.summary_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                existing_rows = list(reader)

        if not existing_rows:
            existing_rows = [SUMMARY_HEADER]

        existing_rows.append(row_values)

        # Write to temporary file in same directory then atomically replace
        summary_dir = os.path.dirname(os.path.abspath(self.summary_path))
        temp_fd, temp_path = tempfile.mkstemp(dir=summary_dir, suffix=".tmp")
        with os.fdopen(temp_fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(existing_rows)

        os.replace(temp_path, self.summary_path)

    def save_queries_parquet(self, run_id: str, query_records: list[dict[str, Any]]) -> str:
        """Save detailed per-question measurements to Parquet."""
        out_dir = f"results/runs/{run_id}"
        os.makedirs(out_dir, exist_ok=True)
        parquet_path = f"{out_dir}/queries.parquet"
        df = pd.DataFrame(query_records)
        df.to_parquet(parquet_path, index=False, engine="pyarrow")
        return parquet_path
