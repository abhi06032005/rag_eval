"""Audit contracts for committed experiment evidence, distinct from mock unit fixtures."""

import gzip
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path("results/recomputed")


@pytest.mark.parametrize("dataset", ["scifact", "nfcorpus", "fiqa"])
def test_new_rankings_have_complete_matched_populations(dataset):
    root = ROOT / dataset
    if not (root / "per_query.jsonl.gz").exists():
        pytest.skip("Independent experiment not yet run")
    qids = set(json.loads((root / "query_ids.json").read_text()))
    corpus = set(json.loads((root / "corpus_ids.json").read_text()))
    groups = {}
    with gzip.open(root / "per_query.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = (row["depth"], row["system"])
            group = groups.setdefault(key, set())
            assert row["query_id"] not in group
            group.add(row["query_id"])
            ids = [d for d, s in row["rankings"]]
            assert len(ids) == len(set(ids)) and set(ids) <= corpus
            assert len(ids) <= row["depth"]
            assert all(0 <= v <= 1.0000000001 for v in row["metrics"].values())
            assert row["metrics"]["Recall@1"] <= row["metrics"]["Recall@5"] <= row["metrics"]["Recall@10"]
    assert len(groups) == 12
    assert all(group == qids for group in groups.values())
    stats = json.loads((root / "statistics.json").read_text())
    assert len(stats) == 36
    assert all(s["n"] == len(qids) and s["p_holm"] + 0.00005 >= s["p_centered_bootstrap"] for s in stats)


@pytest.mark.parametrize("dataset", ["scifact", "nfcorpus", "fiqa"])
def test_replay_fixed_router_matches_executed_rankings(dataset):
    root = ROOT / dataset
    if not (root / "ablations_per_query.jsonl.gz").exists():
        pytest.skip("Replay not yet run")
    executed = {}
    with gzip.open(root / "per_query.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["system"] == "Adaptive_Router" and r["depth"] in (50, 250):
                executed[r["depth"], r["query_id"]] = r
    replayed = set()
    with gzip.open(root / "ablations_per_query.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["system"] == "Router_fixed_defaults":
                key = (r["depth"], r["query_id"])
                old = executed[key]
                assert r["rankings"] == old["rankings"][:10]
                assert r["route"] == old["route"]
                assert r["metrics"]["nDCG@10"] == old["metrics"]["nDCG@10"]
                replayed.add(key)
    assert replayed == set(executed)


def test_corrected_registry_uses_250_branch_candidates_for_legacy_d50():
    registry = json.loads(Path("results/canonical_results_registry.json").read_text())
    if registry.get("registry_version") != "2.0-evidence-derived":
        pytest.skip("Registry not regenerated")
    record = next(r for r in registry["records"] if r["experiment_id"] == "EXP-SCIFACT-HYBRID-D50")
    assert record["retrieval_depth_k_cand"] == 250 and record["retrieval_top_k"] == 50
    raw = Path(record["rankings_path"])
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == record["rankings_sha256"]
    summary = json.loads((ROOT / "scifact_reconciliation/summary.json").read_text())["metrics"]
    assert record["nDCG@10"] == pytest.approx(summary["reconstructed_branch_D250"]["nDCG@10"])
    assert record["Recall@10"] == pytest.approx(summary["reconstructed_branch_D250"]["Recall@10"])
    assert record["nDCG@10"] != pytest.approx(summary["reconstructed_branch_D50"]["nDCG@10"])
