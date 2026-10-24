"""Offline audit of committed new-run outputs. Not a substitute for rerunning BEIR."""
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COUNTS = {"scifact": (300, 5183), "nfcorpus": (323, 3633), "fiqa": (648, 57638)}
DEPTHS = (50, 100, 250)
SYSTEMS = {"BM25", "Dense_Exact", "Fixed_Hybrid", "Adaptive_Router"}
METRICS = ("Recall@1", "Recall@5", "Recall@10", "nDCG@10", "MRR@10", "Precision@10")
TOL = 1e-9


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def close(actual, expected, name):
    if abs(float(actual) - float(expected)) > TOL:
        raise AssertionError(f"{name}: {actual} != {expected}")


def validate():
    registry = read("results/canonical_results_registry.json")
    mapped = {row["experiment_id"]: row for row in registry["records"]}
    total = 0
    for ds, (expected_queries, expected_docs) in COUNTS.items():
        summaries = read(f"results/recomputed/{ds}/metrics.json")
        query_ids = read(f"results/recomputed/{ds}/query_ids.json")
        assert len(query_ids) == expected_queries, (ds, len(query_ids))
        assert len(set(query_ids)) == expected_queries, ds
        assert len(summaries) == len(DEPTHS) * len(SYSTEMS), ds
        rows = {(r["candidate_depth_per_branch"], r["system"]): r for r in summaries}
        assert set(rows) == {(d, s) for d in DEPTHS for s in SYSTEMS}, ds
        for (depth, system), row in rows.items():
            assert row["n"] == expected_queries, (ds, system)
            assert row["returned_depth"] == depth, (ds, system)
            for metric in METRICS:
                v = row["metrics"][metric]
                assert 0 <= v <= 1, (ds, depth, system, metric, v)
            ref = mapped[f"REEXEC-{ds.upper()}-{system}-CAND{depth}"]
            assert ref["num_docs"] == expected_docs
            assert ref["num_queries"] == expected_queries
            for metric in METRICS:
                close(ref[metric], row["metrics"][metric], (ds, depth, system, metric))
        # Independently recompute means of saved per-query metrics.
        sums = defaultdict(lambda: defaultdict(float))
        seen = defaultdict(set)
        counter = 0
        with gzip.open(ROOT / f"results/recomputed/{ds}/per_query.jsonl.gz", "rt", encoding="utf-8") as file:
            for line in file:
                obj = json.loads(line)
                k = (obj["depth"], obj["system"])
                assert k in rows, k
                assert obj["query_id"] in query_ids
                assert obj["query_id"] not in seen[k], (ds, k, obj["query_id"])
                seen[k].add(obj["query_id"])
                for metric in METRICS:
                    sums[k][metric] += float(obj["metrics"][metric])
                counter += 1
        assert counter == expected_queries * len(DEPTHS) * len(SYSTEMS), (ds, counter)
        for k, row in rows.items():
            assert len(seen[k]) == expected_queries, (ds, k)
            for metric in METRICS:
                close(sums[k][metric] / expected_queries, row["metrics"][metric],
                      (ds, k, metric))
        total += counter
        print(f"PASS {ds}: {expected_queries} queries x {len(DEPTHS)} depths x 4 systems")
    summary = read("results/recomputed/scifact_reconciliation/summary.json")["metrics"]
    legacy = mapped["EXP-SCIFACT-HYBRID-D50"]
    for metric in ("Recall@10", "nDCG@10", "MRR@10"):
        close(legacy[metric], summary["reconstructed_branch_D250"][metric], metric)
    status = read("results/recomputed/status.json")
    assert status["overall_full_objective"] == "BLOCKED"
    assert status["new_end_to_end_generation"] == "BLOCKED"
    assert status["default_turbopack_build"] == "FAIL"
    assert status["webpack_production_build"] == "PASS"
    print(f"PASS registry/per-query consistency: {total} saved evaluation rows")
    print("NOTE: archived neural generation and raw BEIR dataset provenance not rerun in CI")


if __name__ == "__main__":
    validate()
