"""
Master Benchmark Execution Script for RAGScaleBench.
Executes controlled baseline experiments and optimization sweeps without fabrication.
Baseline A: Exact Vector Search
Baseline B: Default Qdrant HNSW
Baseline C: Optimised HNSW (m=32, ef_c=128, hnsw_ef=128)
Baseline D: Scalar Quantisation (INT8)
Baseline E: Binary Quantisation (1-bit)
Optimisation 1: Fast Query-Time Search (hnsw_ef=32)
Optimisation 2: Fine-Grained 128-token Chunking
Optimisation 3: Coarse 448-token Chunking
"""

import argparse
import os
import sys
import pandas as pd

from ragbench.config import BenchmarkConfig
from ragbench.experiments.runner import load_qa_records, run_single_experiment
from ragbench.experiments.state import ExperimentStateManager


def run_all_benchmarks(
    corpus_parquet: str = "data/processed/documents.parquet",
    dev_qa_path: str = "data/qa/dev.jsonl",
    test_qa_path: str = "data/qa/test.jsonl",
    summary_csv: str = "results/summary.csv",
    evaluate_generation: bool = True,
    max_docs: int | None = None,
) -> None:
    if not os.path.exists(corpus_parquet):
        print(f"Error: Corpus Parquet not found at {corpus_parquet}", file=sys.stderr)
        sys.exit(1)

    docs_df = pd.read_parquet(corpus_parquet)
    if max_docs:
        docs_df = docs_df.head(max_docs)
    print(f"Loaded {len(docs_df)} documents from {corpus_parquet}")

    dev_qa = load_qa_records(dev_qa_path)
    test_qa = load_qa_records(test_qa_path)
    print(f"Loaded {len(dev_qa)} dev questions and {len(test_qa)} test questions")

    state_mgr = ExperimentStateManager(summary_csv)

    # 1. Baseline A: Exact Vector Search Ground Truth
    base_a = BenchmarkConfig(
        embedding_model="BAAI/bge-small-en-v1.5",
        chunk_size_tokens=256,
        overlap_tokens=32,
        hnsw_m=16,
        ef_construct=128,
        hnsw_ef=64,
        quantization="none",
        exact=True,
        generator_backend="mock",
    )

    # 2. Baseline B: Default Qdrant HNSW Approximate Search
    base_b = BenchmarkConfig(
        embedding_model="BAAI/bge-small-en-v1.5",
        chunk_size_tokens=256,
        overlap_tokens=32,
        hnsw_m=16,
        ef_construct=128,
        hnsw_ef=64,
        quantization="none",
        exact=False,
        generator_backend="mock",
    )

    # 3. Baseline C: Optimised HNSW Configuration (Higher graph degree & search depth)
    base_c = BenchmarkConfig(
        embedding_model="BAAI/bge-small-en-v1.5",
        chunk_size_tokens=256,
        overlap_tokens=32,
        hnsw_m=32,
        ef_construct=128,
        hnsw_ef=128,
        quantization="none",
        exact=False,
        generator_backend="mock",
    )

    # 4. Baseline D: Scalar Quantisation (INT8 compression)
    base_d = BenchmarkConfig(
        embedding_model="BAAI/bge-small-en-v1.5",
        chunk_size_tokens=256,
        overlap_tokens=32,
        hnsw_m=16,
        ef_construct=128,
        hnsw_ef=64,
        quantization="scalar",
        exact=False,
        generator_backend="mock",
    )

    # 5. Baseline E: Binary Quantisation (1-bit aggressive compression)
    base_e = BenchmarkConfig(
        embedding_model="BAAI/bge-small-en-v1.5",
        chunk_size_tokens=256,
        overlap_tokens=32,
        hnsw_m=16,
        ef_construct=128,
        hnsw_ef=64,
        quantization="binary",
        exact=False,
        generator_backend="mock",
    )

    # 6. Optimisation 1: Fast Query-Time Search (Lower candidate breadth)
    opt_1 = BenchmarkConfig(
        embedding_model="BAAI/bge-small-en-v1.5",
        chunk_size_tokens=256,
        overlap_tokens=32,
        hnsw_m=16,
        ef_construct=128,
        hnsw_ef=32,
        quantization="none",
        exact=False,
        generator_backend="mock",
    )

    # 7. Optimisation 2: Fine-Grained 128-token Chunking
    opt_2 = BenchmarkConfig(
        embedding_model="BAAI/bge-small-en-v1.5",
        chunk_size_tokens=128,
        overlap_tokens=0,
        hnsw_m=16,
        ef_construct=128,
        hnsw_ef=64,
        quantization="none",
        exact=False,
        generator_backend="mock",
    )

    # 8. Optimisation 3: Coarse 448-token Chunking
    opt_3 = BenchmarkConfig(
        embedding_model="BAAI/bge-small-en-v1.5",
        chunk_size_tokens=448,
        overlap_tokens=32,
        hnsw_m=16,
        ef_construct=128,
        hnsw_ef=64,
        quantization="none",
        exact=False,
        generator_backend="mock",
    )

    experiment_suite = [
        ("Baseline A (Exact Search)", base_a),
        ("Baseline B (Default HNSW)", base_b),
        ("Baseline C (Optimised HNSW m=32, ef=128)", base_c),
        ("Baseline D (Scalar Quantisation)", base_d),
        ("Baseline E (Binary Quantisation)", base_e),
        ("Optimisation 1 (Fast Search ef=32)", opt_1),
        ("Optimisation 2 (Fine Chunking 128/0)", opt_2),
        ("Optimisation 3 (Coarse Chunking 448/32)", opt_3),
    ]

    print("\n========================================================")
    print(f"Starting Execution of {len(experiment_suite)} Controlled Benchmark Experiments")
    print("========================================================")

    for name, cfg in experiment_suite:
        print(f"\n>>> Running: {name}")
        run_single_experiment(
            cfg=cfg,
            documents_df=docs_df,
            dev_qa=dev_qa,
            test_qa=test_qa,
            state_mgr=state_mgr,
            evaluate_generation=evaluate_generation,
        )

    print("\n========================================================")
    print("All benchmark experiments completed successfully!")
    print(f"Results recorded in: {summary_csv}")
    print("========================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run controlled RAGScaleBench experiments.")
    parser.add_argument("--max-docs", type=int, default=None, help="Limit documents for quick run")
    parser.add_argument("--summary", type=str, default="results/summary.csv", help="Summary CSV")
    args = parser.parse_args()

    run_all_benchmarks(max_docs=args.max_docs, summary_csv=args.summary)
