"""Render the scientific report exclusively from saved evidence; requires completed experiments."""

import json
from pathlib import Path

from scripts.full_beir_reevaluation import OUT, dump, sha

DATASETS = ("scifact", "nfcorpus", "fiqa")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def table(headers, rows):
    return (
        "\n".join(
            ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
            + ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
        )
        + "\n"
    )


def fmt(x):
    return f"{x:.6f}" if isinstance(x, float) else str(x)


def main():
    manifests = read(OUT / "dataset_verification.json")
    metrics = {n: read(OUT / n / "metrics.json") for n in DATASETS}
    builds = {n: read(OUT / n / "build.json") for n in DATASETS}
    ann = {n: read(OUT / n / "qdrant/metrics.json") for n in DATASETS}
    ablations = {n: read(OUT / n / "ablations.json") for n in DATASETS}
    discrepancy = read(OUT / "scifact_reconciliation/summary.json")
    generation = read(OUT / "generation_audit.json")
    sections = [
        """# Full BEIR Reevaluation

Execution date: 2026-10-24 (Asia/Kolkata). Repository baseline: `1f9dad3`. The evaluation-matrix report reviewed came from remote feature commit `3137b2212da27eab40f26f0edad31a2a9daf0def`, not main; its exact text is retained in [the archived input](../results/recomputed/ARCHIVED_EVALUATION_MATRIX_REPORT.md).

## Verdict and evidence boundaries

**Retrieval reproduction and SciFact reconciliation: PASS. Full end-to-end RAG scientific evaluation: BLOCKED.** No new neural answers or validated claim-level judgments are claimed. Historical registry integrity and the default Turbopack build have failures documented below; an alternate production webpack build passed.

This report separates (1) newly encoded datasets and newly executed retrieval/server experiments, (2) recomputation/replay from saved ranked outputs, (3) archived summaries, and (4) mock fixtures and unverified claims. “PASS” means the specified checks completed, not that a system outperforms another or is production-ready.

## Official dataset verification and leakage

Archives were freshly acquired over verified HTTPS from the official BEIR distribution, matched against BEIR's published MD5 values, and compared with the local archives. SHA-256 digests were independently calculated, matched between downloaded/local bytes, and pinned in the acquisition script; they are **not upstream-published SHA-256 values**. Every extracted corpus/query/qrels file was byte-checked against its archive. IDs are unique, qrel references resolve, and no benchmark labels were generated or substituted.

Sources: [BEIR dataset catalog](https://github.com/beir-cellar/beir), [SciFact card](https://huggingface.co/datasets/BeIR/scifact), [NFCorpus card](https://huggingface.co/datasets/BeIR/nfcorpus), [FiQA card](https://huggingface.co/datasets/BeIR/fiqa). All three BEIR cards declare CC BY-SA 4.0; this records their declarations, not a legal audit of underlying source material. See [DATA_LICENSES.md](../DATA_LICENSES.md).
"""
    ]
    sections.append(
        table(
            [
                "Dataset",
                "Corpus docs",
                "All query records",
                "Train / dev / test query counts",
                "Test qrel rows",
                "Test grades",
            ],
            [
                [
                    n,
                    m["documents"],
                    m["queries"],
                    " / ".join(str(m["splits"].get(s, {}).get("queries", "absent")) for s in ("train", "dev", "test")),
                    m["splits"]["test"]["judgments"],
                    m["splits"]["test"]["grades"],
                ]
                for n, m in manifests.items()
            ],
        )
    )
    sections.append(table(["Dataset", "Archive SHA-256"], [[n, m["archive_sha256"]] for n, m in manifests.items()]))
    overlaps = []
    for n, m in manifests.items():
        for pair, r in m["overlap"].items():
            overlaps.append([n, pair, len(r["query_ids"]), len(r["normalized_query_texts"])])
    sections.append(
        table(["Dataset", "Split pair", "Shared query IDs", "Shared normalized exact query texts"], overlaps)
    )
    sections.append("""The shared corpus is intentional in retrieval; document overlap across qrels is not itself query leakage. Exact normalized query text overlap is a real limitation even when IDs differ. SciFact train/test shares two texts; NFCorpus dev/test also shares a text. Calibration replay excludes matching test texts in a separate ablation, without removing test queries. Exact-text checks do not rule out paraphrases, model-pretraining contamination, or undisclosed historical test-set tuning. Full overlap strings and component hashes are in [dataset_verification.json](../results/recomputed/dataset_verification.json).

## Retrieval protocol and metric definitions

All four systems use exactly each dataset's official test-qrels query population: 300 / 323 / 648 queries. Documents are `title + space + text`, with no synthetic corpus expansion or chunk-level substitute labels. Dense embeddings are newly generated with `sentence-transformers/all-MiniLM-L6-v2`, pinned revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, CPU float32, 384 dimensions, L2 normalization, maximum sequence length 256. Document/query batch sizes are 64/32. Model truncation can discard trailing document content.

BM25 is the repository's regex tokenizer and smoothed RSJ variant (`k1=1.5`, `b=0.75`), not an Elasticsearch baseline. Dense exact search exhaustively scores the complete corpus. Equal dense scores use stable corpus input order; the legacy reverse-argsort tie order is separately audited. RRF uses `k=60`. Candidate depths **50, 100, 250 per branch** are explicitly matched; returned depth equals candidate depth. Quality is evaluated at the same top-10 cutoff. The router's main matrix uses fixed defaults (margin 0.28, IDF 6, dense-confidence 0.75, weak-BM25 8), with no development/test fitting. Historical calibration and feature ablations are separate below.

Recall@k counts unique relevant documents / all positive qrels; Precision@k divides by k even when fewer documents are returned. Graded nDCG@10 uses linear judgment gain and log2(rank+1) discount; binary nDCG is retained only as a legacy diagnostic. MRR@10 is the reciprocal first relevant rank, censored at 10. Full retained-depth MRR is separately named MRR@50/100/250 in the raw metric JSON. Unjudged documents have gain zero. Query IDs are not semantically excluded from document IDs; this follows the repository's protocol, and differs from evaluator configurations that automatically exclude identical IDs.

## Fresh retrieval quality: dataset / model / configuration matrix

The model column distinguishes lexical-only BM25 from MiniLM dense retrieval; Hybrid and Router combine BM25 with that same encoder. These results do not establish performance for other embedding models. All values below are newly executed; archived numbers are not inserted into missing cells.
""")
    rows = []
    for n in DATASETS:
        for r in metrics[n]:
            m = r["metrics"]
            rows.append(
                [n, r["system"], r["candidate_depth_per_branch"]]
                + [
                    fmt(m[k])
                    for k in (
                        "Recall@1",
                        "Recall@5",
                        "Recall@10",
                        "nDCG@10",
                        "MRR@10",
                        "Precision@1",
                        "Precision@5",
                        "Precision@10",
                    )
                ]
            )
    sections.append(
        table(
            [
                "Dataset",
                "System",
                "Candidates / returned",
                "R@1",
                "R@5",
                "R@10",
                "graded nDCG@10",
                "MRR@10",
                "P@1",
                "P@5",
                "P@10",
            ],
            rows,
        )
    )
    sections.append("""## SciFact D50 discrepancy: root cause and correction

The old `audit_and_verify_phase4.py` saved only **50 BM25 and 50 dense hits** for each query, although its stored fused hybrid list came from depth 100. The old registry builder then re-fused those saved top-50 branches, called the result the Stage 2 `top_k=50` run, and described 250 candidates without having the missing 200 candidates. The actual Stage 2 policy requests `max(top_k * 5, 50)`: returning 50 therefore means **250 candidates per branch**. The label “D50” conflated returned cutoff with branch depth. Changing the returned cutoff alone cannot change a fixed fused ranking's first ten documents.

We reloaded official qrels, verified all 300 query IDs, recomputed the saved branch outputs, rebuilt branch lists to 250 using archived vectors, and independently repeated retrieval with fresh embeddings. The reconstructed top-50 branches match the archive for every query. Both disputed values are reproducible under their correct configurations:
""")
    sections.append(
        table(
            ["Evidence / configuration", "nDCG@10", "Recall@10", "MRR@10", "MRR@50"],
            [
                [k] + [fmt(v[x]) for x in ("nDCG@10", "Recall@10", "MRR@10", "MRR@50")]
                for k, v in discrepancy["metrics"].items()
            ],
        )
    )
    sections.append("""The corrected `EXP-SCIFACT-HYBRID-D50` entry means **250 branch candidates, 50 returned**, yielding 0.6798740045 nDCG@10 and 0.8103888889 Recall@10. Its MRR@10 is 0.6427830688; the former mixed-depth MRR 0.6452 must not be carried forward. The 0.6851393844 / 0.8253888889 result remains a valid **50-candidate** ablation, not the matched Stage 2 run.

The fix changes the data path, not just the display: `FixedHybridPolicy` and `AdaptiveRouterPolicy` now accept explicit candidate depth; regression tests verify invariant top-10 prefixes across returned cutoffs at fixed depth. The registry builder recomputes quality from ranked documents and qrels, checks population completeness and aggregate consistency, and removes hard-coded measurements and conclusions. The original registry is preserved as [an archived input](../results/recomputed/archived_canonical_results_registry.json). [Reconciliation evidence](../results/recomputed/scifact_reconciliation/summary.json) includes hashes and [ranked outputs](../results/recomputed/scifact_reconciliation/per_query.jsonl.gz).

## Other inconsistencies and calibrated-router replication

NFCorpus has relevance grades 1 and 2; the earlier scripts converted them into sets, reporting binary nDCG as nDCG without qualification. The new main matrix preserves grades. The legacy NFCorpus Dense Exact binary nDCG 0.3181067291 is reproducible with its reverse-argsort tie ordering; stable tie ordering gives 0.3184208417, while graded stable nDCG is 0.3166731275. The drift audit isolates tie ordering from tiny query-embedding floating-point differences. Newly rebuilt SciFact and NFCorpus corpus vectors exactly match the archived corpus vectors.

The original FiQA calibration function slices `dev_qids[:300]`, despite a prior report saying all 500 development queries were used. NFCorpus's actual calibration selects margin 0.15 / IDF 4.0, not the hard-coded registry description 0.25 / 6.0. Calibration is reproduced directly from the original function bodies, with an additional exact-text-deduplicated variant. These are exploratory replications, not preregistered selection or equivalence trials.
""")
    rows = []
    for n in DATASETS:
        for r in ablations[n]:
            if r["depth"] == 250 and r["system"] in (
                "Router_historical",
                "Router_deduplicated_calibration",
                "Router_fixed_defaults",
            ):
                rows.append(
                    [
                        n,
                        r["system"],
                        fmt(r["metrics"]["nDCG@10"]),
                        fmt(r["metrics"]["binary_nDCG@10"]),
                        fmt(r["metrics"]["Recall@10"]),
                        f"{r['lexical_routes']}/{r['n']}",
                    ]
                )
    sections.append(
        table(
            [
                "Dataset",
                "250-candidate router",
                "graded nDCG@10",
                "binary nDCG@10",
                "Recall@10",
                "Lexical routes (replay)",
            ],
            rows,
        )
    )
    sections.append("""Historical SciFact calibration reproduces 0.6861231432 nDCG@10 and 138/300 lexical-only routes. The main fixed-default matrix is a different router configuration and must not be silently substituted for it. Replay lexical routes describe search decisions; all query embeddings had already been computed, so these counts do not measure avoided neural computation in the fresh run.

## Paired inference

Statistics use query-aligned paired differences, 10,000 bootstrap resamples, seed 42, percentile 95% confidence intervals, and a two-sided null-centered bootstrap p-value with a +1 finite-resample correction. All test queries are required on both sides. Holm (FWER) and Benjamini-Hochberg (FDR) adjustments cover **36 comparisons per dataset** in the main matrix: six system pairs × three depths × two metrics. Calibrated-router replay has its own 12-comparison family; ANN has a separate family of all exact/ANN configuration pairs × two metrics. Corrections are not global across those families or datasets. Tests are descriptive/exploratory. No non-inferiority or equivalence claim is made; a nonsignificant result is not proof of equality.

Selected matched 250-candidate main comparisons follow. Complete results for both nDCG@10 and Recall@10 are in each dataset's `statistics.json`.
""")
    rows = []
    for n in DATASETS:
        for s in read(OUT / n / "statistics.json"):
            if s["depth"] == 250 and s["metric"] == "nDCG@10":
                rows.append(
                    [
                        n,
                        s["a"] + " - " + s["b"],
                        fmt(s["delta_a_minus_b"]),
                        f"[{s['ci95'][0]:.6f}, {s['ci95'][1]:.6f}]",
                        fmt(s["p_centered_bootstrap"]),
                        fmt(s["p_holm"]),
                        fmt(s["q_bh"]),
                    ]
                )
    sections.append(table(["Dataset", "Pair", "Delta", "95% CI", "Raw p", "Holm p", "BH q"], rows))
    sections.append("""## Fresh search timing, memory and index build measurements

Search timing is one serial pass per configuration, using **precomputed query vectors**. It excludes query encoding, generation and network for the NumPy/BM25 matrix. QPS is inverse mean measured search time, not load-tested service capacity. Qdrant timing includes local HTTP overhead, with a warm-up query. Runs shared this laptop with other audit work (including embedding/build processes); these are observational timings, **not isolated performance comparisons**. No thermal, energy, GPU, concurrency-capacity or latency-improvement claim is made. p99 is based on only 300/323/648 queries and has limited tail precision.

RSS values are process-wide snapshots or process peak working set, not per-index incremental memory. Qdrant RSS includes collections retained from previous configurations. Index sizes are actual serialized lexical index, float32 array file, or full Qdrant collection-directory bytes (including storage overhead/WAL), not theoretical quantization ratios.
""")
    sections.append(
        table(
            [
                "Dataset",
                "Corpus encoding s",
                "Query encoding s",
                "BM25 build s",
                "Dense array bytes",
                "BM25 index bytes",
                "Process peak working set bytes",
            ],
            [
                [
                    n,
                    fmt(b["corpus_encoding_seconds"]),
                    fmt(b["query_encoding_seconds"]),
                    fmt(b["bm25_build_seconds"]),
                    b["corpus_vectors_bytes"],
                    b["bm25_pickle_bytes"],
                    b["process_peak_wset_bytes"],
                ]
                for n, b in builds.items()
            ],
        )
    )
    sections.append(
        table(
            ["Dataset", "System", "Depth", "p50 ms", "p95 ms", "p99 ms", "Serial search QPS", "Process RSS bytes"],
            [
                [
                    n,
                    r["system"],
                    r["candidate_depth_per_branch"],
                    fmt(r["p50_ms"]),
                    fmt(r["p95_ms"]),
                    fmt(r["p99_ms"]),
                    fmt(r["serial_search_qps"]),
                    r["process_rss_bytes"],
                ]
                for n in DATASETS
                for r in metrics[n]
            ],
        )
    )
    sections.append("""## Real Qdrant HNSW and quantization

Server: Qdrant 1.19.2, isolated HTTP port 6335, CPU indexing. Each collection is required to be GREEN with its **full corpus indexed** before evaluation. No in-memory client fallback is used. Standard HNSW uses M=16 / ef_construct=100; tuned and quantized variants use M=32 / ef_construct=200. HNSW ef is 32 or 128; INT8 uses quantile 0.99; binary uses 1-bit quantization. Quantization rescore is ablated at ef=128, oversampling=2. Index/full-scan thresholds are 10 KB (the server rejected the first attempt's value 1). HNSW construction has no exposed fixed random seed here; exact reruns can differ. The [Qdrant quantization documentation](https://qdrant.tech/documentation/manage-data/quantization/) describes the approximation/rescoring tradeoff.

ANN Recall@10 is intersection with freshly computed exact dense top-10 / 10, distinct from relevance-judgment Recall@10. Equal-distance neighbors can affect exact-set overlap. Rescoring cannot recover neighbors absent from the retrieved candidate pool. These settings are not assumed identical to archived ANN experiments.
""")
    sections.append(
        table(
            [
                "Dataset",
                "Configuration",
                "ANN R@10",
                "Gold R@10",
                "graded nDCG@10",
                "p50 ms",
                "p95 ms",
                "p99 ms",
                "Serial HTTP QPS",
                "Build/upload/optimize s",
                "Collection bytes",
                "Server RSS bytes",
            ],
            [
                [
                    n,
                    r["configuration"],
                    fmt(r["ANN_Recall@10"]),
                    fmt(r["metrics"]["Recall@10"]),
                    fmt(r["metrics"]["nDCG@10"]),
                    fmt(r["p50_ms"]),
                    fmt(r["p95_ms"]),
                    fmt(r["p99_ms"]),
                    fmt(r["serial_http_qps"]),
                    fmt(r["build_upload_and_optimize_seconds"]),
                    r["collection_disk_bytes"],
                    r["server_process_rss_bytes"],
                ]
                for n in DATASETS
                for r in ann[n]
            ],
        )
    )
    sections.append("""Other ANN quality metrics (Recall@1/5, Precision@1/5/10 and MRR@10), paired statistics, per-query ranks, and actual collection configurations are retained under each dataset's `qdrant/` directory. Binary quantization's substantial ANN distortion under these settings is a negative result, not evidence of universal compression benefits.

## Fusion and router ablations

These are recomputed from the **new** BM25/dense candidate lists, not new timing measurements. Candidate-depth changes appear in the main matrix; the matrix below additionally varies RRF constant and removes margin/IDF gates or forces hybrid fallback. Fixed-default feature ablations do not inherit historically calibrated thresholds. No significance is inferred merely from these point differences.
""")
    sections.append(
        table(
            ["Dataset", "Depth", "Ablation", "graded nDCG@10", "Recall@10", "MRR@10", "Lexical routes"],
            [
                [
                    n,
                    r["depth"],
                    r["system"],
                    fmt(r["metrics"]["nDCG@10"]),
                    fmt(r["metrics"]["Recall@10"]),
                    fmt(r["metrics"]["MRR@10"]),
                    r["lexical_routes"],
                ]
                for n in DATASETS
                for r in ablations[n]
            ],
        )
    )
    sections.append("""## Generation, citations, faithfulness and abstention

**New end-to-end RAG generation: BLOCKED.** The local Ollama endpoint refuses connections and Docker's daemon is unavailable (captured in `services.json` and `docker-version.txt`). In addition, BEIR qrels are relevance judgments, not reference answers, claim-level entailment annotations or answerability labels. A validated answer/claim test set and a specified, validated judge or documented human annotations are missing. No neural or human judgments were invented to fill these cells.

Inspection of `evaluate_real_llm_rag.py` shows direct calls to the generator with script-supplied contexts. It does not execute retrieval. Therefore the previous report's “end-to-end” characterization is unsupported. The mock-generation suite is a software fixture, not neural evidence. The archived Qwen outputs can support lexical/format recomputation only:
""")
    sections.append(
        table(
            [
                "Evidence",
                "Cases",
                "Answer quality",
                "Citation support",
                "Faithfulness",
                "Abstention",
                "Unsupported-claim rate",
            ],
            [
                ["New retrieval → neural generation", "0", "BLOCKED", "BLOCKED", "BLOCKED", "BLOCKED", "BLOCKED"],
                [
                    "Archived qwen2.5:0.5b supplied-context outputs",
                    generation["archived_cases"],
                    f"Token F1 {generation['token_f1_against_script_authored_reference']:.6f} on {generation['answerable_cases']} script-reference cases",
                    "No bracket citations on 5 answerable cases; entailment unverified",
                    "UNVERIFIED",
                    "2/2 no-context fixtures detected; not general answerability",
                    "UNVERIFIED",
                ],
                [
                    "Mock generator / hashed embeddings",
                    "Software tests only",
                    "Not benchmark evidence",
                    "Not benchmark evidence",
                    "Not benchmark evidence",
                    "Fixture behavior only",
                    "Not benchmark evidence",
                ],
            ],
        )
    )
    sections.append("""Citation-ID presence is not citation entailment. The archive's abstention examples exercise a no-context fast path, not necessarily model reasoning. Script-source-ID checks and per-case lexical recomputations are in [generation_audit.json](../results/recomputed/generation_audit.json). No accuracy, human review, security robustness, or claim-support percentage is inferred from unsupported assertions. No confusion matrices are generated: rankings are not classification predictions, and no validated classification evaluation has been supplied.

## Automated checks and production build

- **PASS:** unit/integration suite, 68 passed, 3 warnings. Tests explicitly use offline hashed embeddings and mock generation; local Qdrant fixtures use exact emulation. These do not validate ANN performance or neural generation. [JUnit](../results/recomputed/pytest-retry.xml), [full output](../results/recomputed/pytest-retry-output.txt).
- **FAIL / environment:** default `npm run build`: restricted font fetching first failed; the unsandboxed retry failed to start Turbopack's CSS worker. [Initial output](../results/recomputed/web-build-output.txt), [retry](../results/recomputed/web-build-retry-output.txt).
- **PASS:** `npm run build -- --webpack`: production compilation, TypeScript and static prerendering completed. [Output](../results/recomputed/web-webpack-output.txt).
- Initial restricted pytest attempts were interrupted; the captured diagnostic run had tokenizer/temp-access failures and an event loop stuck creating a local socket pair. They are not reported as completed test suites. An unrestricted retry passed. Subsequent audit-contract checks, if present, are separately recorded rather than replacing this history.

Exact commands, versions, model revision, hardware and seeds are in [REPRODUCTION_ENVIRONMENT.md](REPRODUCTION_ENVIRONMENT.md). Running a test suite or web build does not verify historical scientific claims.

## Reproducible evidence map and remaining limitations

- [Dataset manifests, archives' hashes and split leakage](../results/recomputed/dataset_verification.json)
- [SciFact discrepancy reconstruction](../results/recomputed/scifact_reconciliation/summary.json)
- [Corrected evidence-derived registry](../results/canonical_results_registry.json)
- [SciFact outputs](../results/recomputed/scifact/), [NFCorpus outputs](../results/recomputed/nfcorpus/), [FiQA outputs](../results/recomputed/fiqa/): `per_query.jsonl.gz`, `metrics.json`, `statistics.json`, `build.json`, calibration and ablation files, and real Qdrant outputs.
- [Generation audit](../results/recomputed/generation_audit.json) and [source/artifact hashes](../results/recomputed/artifact_manifest.json)

Large embedding arrays, index binaries, downloaded archives and Qdrant storage are generated locally and excluded from Git. Rankings, hashes, configuration and build commands are committed. Archived embedding-cache forensic reconstruction is separately labeled; the independent fresh runs regenerate the model inputs. No additional embedding models, GPU, production concurrency/load sweep, or cold-start repetitions are claimed. Historical load figures remain archived and were not remeasured. New end-to-end RAG answer quality, citation entailment, faithfulness, abstention generalization and unsupported-claim rate remain blocked on the dependencies above.
""")
    Path("reports/FULL_BEIR_REEVALUATION.md").write_text("\n".join(sections), encoding="utf-8")
    dump(
        OUT / "status.json",
        {
            "official_data_verification": "PASS",
            "fresh_retrieval": "PASS",
            "scifact_discrepancy": "PASS",
            "real_qdrant": "PASS",
            "unit_integration_tests": "PASS",
            "default_turbopack_build": "FAIL",
            "webpack_production_build": "PASS",
            "new_end_to_end_generation": "BLOCKED",
            "valid_claim_level_labels_and_judge": "BLOCKED",
            "historical_claim_integrity": "FAIL",
            "overall_full_objective": "BLOCKED",
        },
    )


if __name__ == "__main__":
    main()
