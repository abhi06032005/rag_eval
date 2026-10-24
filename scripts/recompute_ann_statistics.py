"""Summarize real ANN outputs with paired statistics against exact dense retrieval."""

import argparse
import gzip
import json

from scripts.full_beir_reevaluation import OUT, dump, statistics


def main(name):
    root = OUT / name
    rows = []
    with gzip.open(root / "qdrant/per_query.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append({"query_id": r["query_id"], "depth": 10, "system": r["configuration"], "metrics": r["metrics"]})
    with gzip.open(root / "per_query.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["depth"] == 250 and r["system"] == "Dense_Exact":
                rows.append({**r, "depth": 10})
    dump(root / "qdrant/statistics.json", statistics(rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    main(p.parse_args().dataset)
