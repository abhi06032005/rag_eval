"""Audit committed historical RAG benchmark artifacts; do not mislabel as fresh trials.

Python 3.11+ stdlib only. Regenerates reports/EVALUATION_MATRIX_REPORT.md
and results/evaluation_matrix_report.json. --check detects stale files.
"""
import argparse
import csv
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/EVALUATION_MATRIX_REPORT.md"
OUTPUT = ROOT / "results/evaluation_matrix_report.json"
DATASET_COUNTS = {"scifact": (5183, 300), "nfcorpus": (3633, 323), "fiqa": (57638, 648)}
SYSTEMS = ["BM25", "Dense_Exact", "Fixed_Hybrid", "Adaptive_Router"]
METRICS = ["nDCG@10", "Recall@10", "MRR"]


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def read_csv(path):
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def n(v):
    return None if v is None or v == "" else round(float(v), 4)


def stat(ds, a, b, metric="ndcg10"):
    obj = read_json(f"results/{ds}/statistical_validation.json")
    return obj[f"{a}_vs_{b}_{metric}"]


def datasets_and_issues():
    registry = read_json("results/canonical_results_registry.json")
    records = {r["experiment_id"]: r for r in registry["records"]}
    ids = {
        "BM25": "EXP-SCIFACT-BM25",
        "Dense_Exact": "EXP-SCIFACT-DENSE-EXACT",
        "Fixed_Hybrid": "EXP-SCIFACT-HYBRID-D50",
        "Adaptive_Router": "EXP-SCIFACT-ADAPT-D50",
    }
    scifact = {k: {m: n(records[v].get(m)) for m in METRICS} for k, v in ids.items()}
    issues = []
    for metric, suffix in [("nDCG@10", "ndcg10"), ("Recall@10", "recall10")]:
        matched = stat("scifact", "Fixed_Hybrid", "Adaptive_Router", suffix)
        actual = scifact["Fixed_Hybrid"][metric]
        paired = n(matched["mean_a"])
        if abs(actual - paired) > 0.0003:
            issues.append(f"SciFact D50 Fixed Hybrid {metric}: registry={actual:.4f}, matched bootstrap={paired:.4f}; unresolved until source rankings are independently re-evaluated.")
        scifact["Fixed_Hybrid"][metric] = paired
        scifact["Adaptive_Router"][metric] = n(matched["mean_b"])
    data = {"scifact": scifact}
    for ds, path, key in [
        ("nfcorpus", "results/nfcorpus/metrics_summary.csv", "Policy"),
        ("fiqa", "results/fiqa/metrics_summary.csv", "Configuration"),
    ]:
        rows = {r[key]: r for r in read_csv(path)}
        data[ds] = {s: {m: n(rows[s][m]) for m in METRICS} for s in SYSTEMS}
    for ds, systems in data.items():
        for sys in systems:
            for metric, val in systems[sys].items():
                if val is not None and not (0 <= val <= 1):
                    raise ValueError(f"Invalid {ds}/{sys}/{metric}={val}")
        matched = stat(ds, "Fixed_Hybrid", "Adaptive_Router")
        for sys, field in [("Fixed_Hybrid", "mean_a"), ("Adaptive_Router", "mean_b")]:
            if abs(data[ds][sys]["nDCG@10"] - n(matched[field])) > 0.0003:
                issues.append(f"{ds}: {sys} nDCG does not match paired bootstrap output")
        if abs(n(matched["mean_a"]) - n(matched["mean_b"]) - float(matched["mean_diff"])) > 0.001:
            issues.append(f"{ds}: bootstrap point estimates do not match stored delta")
    return data, records, issues


def matrix(data, metric):
    out = [
        "| BEIR test dataset (docs/queries) | BM25 | Dense Exact | Fixed Hybrid | Adaptive Router |",
        "|---|---:|---:|---:|---:|",
    ]
    for ds, (docs, q) in DATASET_COUNTS.items():
        cells = ["N/A" if data[ds][s][metric] is None else f"{data[ds][s][metric]:.4f}" for s in SYSTEMS]
        out.append(f"| {ds} ({docs:,}/{q}) | " + " | ".join(cells) + " |")
    return "\n".join(out)


def build():
    data, records, issues = datasets_and_issues()
    comparisons = []
    for ds, a, b in [
        ("scifact", "Fixed_Hybrid", "Adaptive_Router"),
        ("nfcorpus", "Fixed_Hybrid", "Adaptive_Router"),
        ("fiqa", "Dense_Exact", "Fixed_Hybrid"),
        ("fiqa", "Dense_Exact", "Adaptive_Router"),
    ]:
        row = stat(ds, a, b)
        comparisons.append({
            "dataset": ds, "system_a": a, "system_b": b,
            "a": n(row["mean_a"]), "b": n(row["mean_b"]),
            "delta": n(row["mean_diff"]), "ci95": [n(row["ci_lower_95"]), n(row["ci_upper_95"])],
            "p": float(row["p_value"]), "queries": int(row["num_queries"]),
        })
    raw_qa = read_json("results/qa/generation_evaluation_results.json")["mock_fixture_baseline"]
    raw_llm = read_json("results/qa/real_llm_evaluation_results.json")
    assert raw_qa["is_mock"] is True and raw_llm["is_mock"] is False
    assert len(raw_qa["per_query_results"]) == raw_qa["num_evaluated"]
    assert len(raw_llm["per_query_results"]) == raw_llm["total_test_cases"]
    qa = [
        {"generator": raw_qa["model_name"], "mode": "mock", "n": raw_qa["num_evaluated"],
         "f1": raw_qa["metrics"]["mean_answer_correctness_f1"],
         "faithfulness": raw_qa["metrics"]["mean_faithfulness"],
         "citation_support": raw_qa["metrics"]["mean_citation_support"],
         "abstention": raw_qa["metrics"]["abstention_accuracy"],
         "latency_ms": raw_qa["metrics"]["mean_generation_latency_ms"]},
        {"generator": raw_llm["model_label"], "mode": "neural", "n": raw_llm["total_test_cases"],
         "f1": raw_llm["mean_token_f1_answerable"], "faithfulness": None,
         "citation_support": None, "abstention": raw_llm["abstention_accuracy"],
         "latency_ms": raw_llm["mean_generation_latency_ms"]},
    ]
    ann_ids = [
        ("Standard HNSW", "EXP-SCIFACT-QDRANT-HNSW-STD"),
        ("Tuned HNSW", "EXP-SCIFACT-QDRANT-HNSW-TUNED"),
        ("Scalar INT8 + rescore", "EXP-SCIFACT-QDRANT-SQ-INT8"),
        ("Binary 1-bit + rescore", "EXP-SCIFACT-QDRANT-BQ-1BIT"),
    ]
    ann = [{"config": name, "recall10": records[key]["Recall@10"],
            "ann_recall10": records[key]["ANN_Recall@10"], "ndcg10": records[key]["nDCG@10"],
            "p50_ms": records[key]["latency_p50_ms"]} for name, key in ann_ids]
    loads = sorted(read_json("results/load_tests/concurrency_sweep.json").values(), key=lambda r: r["concurrency"])
    runtime_rows = read_json("results/service_performance/adaptive_runtime_audit.json")["summary_comparison"]
    runtime = {r["Metric / Component"]: r for r in runtime_rows}
    record = {
        "report_type": "historical_artifact_audit_not_fresh_full_benchmark",
        "generated": date.today().isoformat(), "source_data_reembedded": False,
        "new_llm_inference": False, "fresh_qdrant_benchmark": False,
        "datasets": data, "comparisons": comparisons, "generation": qa, "ann": ann,
        "load": loads, "runtime": {k: runtime[k] for k in ["Total Latency p50 (ms)", "Total Latency p95 (ms)", "Total Latency p99 (ms)"]},
        "issues": issues,
    }
    lines = [
        "# RAGScaleBench — Evidence-Backed Evaluation Matrix Report", "",
        "**Compiled:** " + record["generated"] + " | **Evidence class:** committed historic artifacts, checked and tabulated in CI.",
        "",
        "> This is NOT a new end-to-end benchmark execution. The clean Git checkout lacks the full BEIR source corpora, precomputed dense embeddings, original physical Qdrant/LLM runs and hardware needed for full independent reproduction. The script recomputes matrices from saved records and checks consistency only.",
        "",
        "## Dataset population and scope", "",
        "BEIR/SciFact: 5,183 documents, 300 test queries. BEIR/NFCorpus: 3,633 documents, 323 test queries. BEIR/FiQA: 57,638 documents, 648 test queries. Aggregate: 66,454 documents and 1,271 test queries across three domains.",
        "",
        "These are information-retrieval ranking tasks, not binary classification. Without document-level thresholded labels/predictions, a binary confusion matrix would be misleading. The legitimate output is a retrieval-performance matrix using nDCG, Recall and MRR.",
        "",
        "## Matrix A — nDCG@10 (higher is better)", "", matrix(data, "nDCG@10"), "",
        "## Matrix B — Recall@10 (higher is better)", "", matrix(data, "Recall@10"), "",
        "## Matrix C — MRR (higher is better)", "", matrix(data, "MRR"), "",
        "For SciFact the D50 Fixed Hybrid nDCG and Recall values use paired-bootstrap records; SciFact MRR is taken from the canonical registry. See the unresolved reconciliation warnings below.",
        "",
        "## Matrix D — Historical paired bootstrap comparisons (nDCG@10)", "",
        "| Dataset | A - B | Mean A | Mean B | Delta | 95% CI of delta | Stored p | Queries |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for x in comparisons:
        lines.append(f"| {x['dataset']} | {x['system_a']} - {x['system_b']} | {x['a']:.4f} | {x['b']:.4f} | {x['delta']:+.4f} | [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] | {x['p']:.4f} | {x['queries']} |")
    lines += [
        "", "These p-values and confidence intervals come from PRE-EXISTING bootstrap artifacts, not a fresh CI bootstrap. Non-significance does not prove equivalence. The project's 0.025 non-inferiority margin was selected post hoc and cannot be called preregistered.",
        "", "## Matrix E — Qdrant ANN distortion on SciFact", "",
        "| Configuration | Gold Recall@10 | ANN Recall@10 vs exact | nDCG@10 | Archived p50 (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for x in ann:
        lines.append(f"| {x['config']} | {x['recall10']:.4f} | {x['ann_recall10']:.4f} | {x['ndcg10']:.4f} | {x['p50_ms']:.2f} |")
    lines += [
        "", "These timings were measured in prior configurations and must not be conflated with HTTP end-to-end latency or freshly measured runner hardware performance.",
        "", "## Matrix F — Archived concurrency sweep", "",
        "| Concurrency | Queries | p50 (ms) | p95 (ms) | p99 (ms) | RPS | Errors |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in loads:
        lines.append(f"| {x['concurrency']} | {x['queries']} | {x['p50_ms']:.2f} | {x['p95_ms']:.2f} | {x['p99_ms']:.2f} | {x['throughput_rps']:.1f} | {x['errors']} |")
    lines += ["", "### SciFact runtime tail latency (archived)", "",
              "| Metric | Fixed Hybrid | Adaptive Router |", "|---|---:|---:|"]
    for k in record["runtime"]:
        x = record["runtime"][k]
        lines.append(f"| {k} | {float(x['Fixed_Hybrid']):.2f} ms | {float(x['Adaptive_Router']):.2f} ms |")
    lines += [
        "", "SciFact adaptive routing historically avoided 138/300 dense calls, yet had WORSE p99 tail latency (101.23 vs 88.55 ms). Do not claim universal latency improvement.",
        "", "## Matrix G — Generation and grounding (noncomparable populations)", "",
        "| Generator | Mode | Cases | Token F1 | Faithfulness | Citation support | Abstention | Mean generation ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for x in qa:
        def show(v):
            return "N/A" if v is None else f"{float(v):.4f}"
        lines.append(f"| {x['generator']} | {x['mode']} | {x['n']} | {show(x['f1'])} | {show(x['faithfulness'])} | {show(x['citation_support'])} | {show(x['abstention'])} | {float(x['latency_ms']):.2f} |")
    lines += [
        "", "Do NOT pool the mock 20-case and neural 10-case results. Neural-level citation support and faithfulness have not been established in this summary. Do not infer generalizable security/safety effectiveness from 10 cases.",
        "", "## Findings and negatives", "",
        "- On historical FiQA results, Dense Exact outperforms Fixed Hybrid (0.3532 vs 0.3300 nDCG@10, stored bootstrap p=0.0296). Hybrid is not universally better.",
        "- On NFCorpus, Fixed Hybrid outperforms Adaptive Router in archived nDCG@10; router accuracy is domain-dependent.",
        "- On SciFact, adaptive routing avoids expensive dense calls, but Stage 2 hybrid headline has an unresolved registry-vs-bootstrap inconsistency.",
        "", "## Reconciliation / limitations", "",
    ]
    lines += ["- **UNRESOLVED** " + x for x in issues]
    lines += [
        "- The former README top-line values disagree with the canonical CSV/paired statistics. The revised README should use clearly labeled archived values, not mixed pipeline depths.",
        "- The full BEIR dataset, cached dense embeddings and physical Qdrant engine are NOT committed; external data retrieval is required to independently rerun retrieval scores, statistical tests and hardware latency.",
        "- Historic load-test results are not CI remeasurements. No scientific claim here is based on freshly generated neural answers.",
        "- Verified from saved records: row counts, score ranges, matched paired comparisons, generation case counts and document/query inventories. This does not verify original data provenance.",
        "", "## Reproduce artifact-level report (offline)", "",
        "Run: python scripts/build_evaluation_matrix_report.py",
        "Verify: python scripts/build_evaluation_matrix_report.py --check",
        "", "Machine-readable companion: results/evaluation_matrix_report.json.",
        "",
    ]
    return "\n".join(lines), json.dumps(record, indent=2) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    opts = parser.parse_args()
    report, obj = build()
    if opts.check:
        assert REPORT.read_text(encoding="utf-8") == report, "Markdown report differs from source artifacts"
        assert OUTPUT.read_text(encoding="utf-8") == obj, "JSON report differs from source artifacts"
        print("PASSED: generated evaluation matrices match archived inputs")
    else:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(report, encoding="utf-8")
        OUTPUT.write_text(obj, encoding="utf-8")
        print(f"Wrote {REPORT} and {OUTPUT}")


if __name__ == "__main__":
    main()
