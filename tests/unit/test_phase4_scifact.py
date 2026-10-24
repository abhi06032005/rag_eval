"""
Unit and Regression Tests for Phase 4: BEIR SciFact Benchmark and Qdrant Audit Logic.
"""

import json
import os
import pytest
from ragbench.evaluation.bm25 import BM25Okapi, tokenize_text
from ragbench.evaluation.retrieval import (
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_recall_at_k,
)
from scripts.run_scifact_benchmark import rrf_fuse
from ragbench.data.govinfo import extract_text_from_bill_xml


def test_tokenize_text():
    tokens = tokenize_text("CRISPR-Cas9 gene editing in Homo sapiens!")
    assert "crispr-cas9" in tokens
    assert "gene" in tokens
    assert "editing" in tokens
    assert "homo" in tokens
    assert "sapiens" in tokens


def test_bm25_ranking_accuracy():
    docs = [
        ("doc_1", "The quick brown fox jumps over the lazy dog."),
        ("doc_2", "CRISPR-Cas9 endonuclease facilitates targeted genomic cleavage."),
        ("doc_3", "Quantum annealing optimizes high-dimensional combinatorial states."),
    ]
    bm25 = BM25Okapi()
    bm25.fit(docs)

    hits = bm25.search("CRISPR genomic cleavage", top_k=2)
    assert len(hits) > 0
    assert hits[0][0] == "doc_2"
    assert hits[0][1] > 0.0


def test_rrf_fusion_behavior():
    ranked_a = [("doc_a", 10.0), ("doc_b", 8.0), ("doc_c", 6.0)]
    ranked_b = [("doc_b", 12.0), ("doc_a", 11.0), ("doc_d", 5.0)]

    fused = rrf_fuse(ranked_a, ranked_b, k_rrf=60, top_k=3)
    fused_ids = [doc_id for doc_id, _ in fused]

    # Both rank 1 and 2 in both lists should be top 2
    assert "doc_a" in fused_ids[:2]
    assert "doc_b" in fused_ids[:2]
    assert len(fused) == 3


def test_scifact_dataset_integrity():
    scifact_dir = "data/beir/scifact"
    assert os.path.exists(os.path.join(scifact_dir, "corpus.jsonl"))
    assert os.path.exists(os.path.join(scifact_dir, "queries.jsonl"))
    assert os.path.exists(os.path.join(scifact_dir, "qrels/test.tsv"))

    # Verify line count
    with open(os.path.join(scifact_dir, "qrels/test.tsv"), "r", encoding="utf-8") as f:
        qrel_lines = [line.strip().split("\t") for line in f if line.strip()]
    assert len(qrel_lines) > 300  # header + 339 judgments


def test_govinfo_xml_extraction():
    sample_xml = b"""<?xml version="1.0"?>
    <bill bill-stage="Introduced-in-House">
      <form>
        <official-title>To establish a national quantum research initiative.</official-title>
      </form>
      <legis-body>
        <section id="H1234">
          <text>Be it enacted by the Senate and House of Representatives of the United States of America in Congress assembled.</text>
        </section>
      </legis-body>
    </bill>
    """
    title, text = extract_text_from_bill_xml(sample_xml)
    assert "quantum research initiative" in title.lower()
    assert "be it enacted" in text.lower()
