import os
import tempfile

import pandas as pd

from ragbench.config import BenchmarkConfig
from ragbench.experiments.state import ExperimentStateManager


def test_config_hashing_stability():
    cfg1 = BenchmarkConfig(embedding_model="BAAI/bge-small-en-v1.5", chunk_size_tokens=256)
    cfg2 = BenchmarkConfig(embedding_model="BAAI/bge-small-en-v1.5", chunk_size_tokens=256)
    cfg3 = BenchmarkConfig(embedding_model="BAAI/bge-small-en-v1.5", chunk_size_tokens=128)

    assert cfg1.config_hash() == cfg2.config_hash()
    assert cfg1.config_hash() != cfg3.config_hash()


def test_atomic_csv_append_and_resume():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "summary.csv")
        mgr = ExperimentStateManager(csv_path)

        assert os.path.exists(csv_path)
        assert len(mgr.get_completed_config_hashes()) == 0

        # Append row 1
        mgr.append_summary_row_atomic({"config_sha": "hash_alpha", "gold_recall_at_10": 0.85})
        assert "hash_alpha" in mgr.get_completed_config_hashes()

        # Append row 2
        mgr.append_summary_row_atomic({"config_sha": "hash_beta", "gold_recall_at_10": 0.92})
        assert "hash_beta" in mgr.get_completed_config_hashes()

        # Check CSV content
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert list(df["config_sha"]) == ["hash_alpha", "hash_beta"]
