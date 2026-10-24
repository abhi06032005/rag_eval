import os
import tempfile

import pandas as pd

from ragbench.config import BenchmarkConfig
from ragbench.experiments.runner import load_qa_records, run_single_experiment
from ragbench.experiments.state import ExperimentStateManager


def test_mini_experiment_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        summary_csv = os.path.join(tmpdir, "summary.csv")
        state_mgr = ExperimentStateManager(summary_csv)

        # 5 fixture documents
        docs_data = [
            {
                "doc_id": f"doc_{i}",
                "text": f"Study {i}: Empirical evaluation of vector indices and graph search.",
                "title": f"Paper {i}",
                "source": "arxiv",
            }
            for i in range(5)
        ]
        docs_df = pd.DataFrame(docs_data)

        # Load real dev QA records
        dev_qa = load_qa_records("data/qa/dev.jsonl")[:3]
        test_qa = load_qa_records("data/qa/test.jsonl")[:3]

        cfg = BenchmarkConfig(
            embedding_model="mock-embedder",
            embedding_dimensions=384,
            chunk_size_tokens=128,
            overlap_tokens=0,
            generator_backend="mock",
            generator_model="mock-generator",
        )

        res = run_single_experiment(
            cfg=cfg,
            documents_df=docs_df,
            dev_qa=dev_qa,
            test_qa=test_qa,
            state_mgr=state_mgr,
            evaluate_generation=True,
        )

        assert res != {}
        assert os.path.exists(summary_csv)
        df = pd.read_csv(summary_csv)
        assert len(df) == 1
        assert "run_id" in df.columns
        assert df["config_sha"].iloc[0] == cfg.config_hash()
