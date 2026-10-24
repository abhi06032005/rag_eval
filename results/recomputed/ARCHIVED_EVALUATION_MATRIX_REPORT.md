# RAGScaleBench — Evidence-Backed Evaluation Matrix Report

**Compiled:** 2026-10-24 | **Evidence class:** committed historic artifacts, checked and tabulated in CI.

> This is NOT a new end-to-end benchmark execution. The clean Git checkout lacks the full BEIR source corpora, precomputed dense embeddings, original physical Qdrant/LLM runs and hardware needed for full independent reproduction. The script recomputes matrices from saved records and checks consistency only.

## Dataset population and scope

BEIR/SciFact: 5,183 documents, 300 test queries. BEIR/NFCorpus: 3,633 documents, 323 test queries. BEIR/FiQA: 57,638 documents, 648 test queries. Aggregate: 66,454 documents and 1,271 test queries across three domains.

These are information-retrieval ranking tasks, not binary classification. Without document-level thresholded labels/predictions, a binary confusion matrix would be misleading. The legitimate output is a retrieval-performance matrix using nDCG, Recall and MRR.

## Matrix A — nDCG@10 (higher is better)

| BEIR test dataset (docs/queries) | BM25 | Dense Exact | Fixed Hybrid | Adaptive Router |
|---|---:|---:|---:|---:|
| scifact (5,183/300) | 0.6432 | 0.6451 | 0.6799 | 0.6861 |
| nfcorpus (3,633/323) | 0.2934 | 0.3181 | 0.3312 | 0.3211 |
| fiqa (57,638/648) | 0.2247 | 0.3532 | 0.3300 | 0.3216 |

## Matrix B — Recall@10 (higher is better)

| BEIR test dataset (docs/queries) | BM25 | Dense Exact | Fixed Hybrid | Adaptive Router |
|---|---:|---:|---:|---:|
| scifact (5,183/300) | 0.7605 | 0.7833 | 0.8104 | 0.8093 |
| nfcorpus (3,633/323) | 0.1396 | 0.1550 | 0.1584 | 0.1545 |
| fiqa (57,638/648) | 0.2843 | 0.4133 | 0.4113 | 0.4006 |

## Matrix C — MRR (higher is better)

| BEIR test dataset (docs/queries) | BM25 | Dense Exact | Fixed Hybrid | Adaptive Router |
|---|---:|---:|---:|---:|
| scifact (5,183/300) | 0.6140 | 0.6110 | 0.6452 | 0.6541 |
| nfcorpus (3,633/323) | 0.5079 | 0.5112 | 0.5442 | 0.5385 |
| fiqa (57,638/648) | 0.2898 | 0.4382 | 0.4085 | 0.4007 |

For SciFact the D50 Fixed Hybrid nDCG and Recall values use paired-bootstrap records; SciFact MRR is taken from the canonical registry. See the unresolved reconciliation warnings below.

## Matrix D — Historical paired bootstrap comparisons (nDCG@10)

| Dataset | A - B | Mean A | Mean B | Delta | 95% CI of delta | Stored p | Queries |
|---|---|---:|---:|---:|---:|---:|---:|
| scifact | Fixed_Hybrid - Adaptive_Router | 0.6799 | 0.6861 | -0.0062 | [-0.0168, +0.0034] | 0.2214 | 300 |
| nfcorpus | Fixed_Hybrid - Adaptive_Router | 0.3312 | 0.3211 | +0.0101 | [+0.0026, +0.0182] | 0.0056 | 323 |
| fiqa | Dense_Exact - Fixed_Hybrid | 0.3532 | 0.3300 | +0.0233 | [+0.0024, +0.0444] | 0.0296 | 648 |
| fiqa | Dense_Exact - Adaptive_Router | 0.3532 | 0.3216 | +0.0316 | [+0.0099, +0.0537] | 0.0034 | 648 |

These p-values and confidence intervals come from PRE-EXISTING bootstrap artifacts, not a fresh CI bootstrap. Non-significance does not prove equivalence. The project's 0.025 non-inferiority margin was selected post hoc and cannot be called preregistered.

## Matrix E — Qdrant ANN distortion on SciFact

| Configuration | Gold Recall@10 | ANN Recall@10 vs exact | nDCG@10 | Archived p50 (ms) |
|---|---:|---:|---:|---:|
| Standard HNSW | 0.7800 | 0.9967 | 0.6417 | 14.54 |
| Tuned HNSW | 0.7833 | 0.9990 | 0.6451 | 16.10 |
| Scalar INT8 + rescore | 0.7833 | 0.9960 | 0.6451 | 23.74 |
| Binary 1-bit + rescore | 0.7817 | 0.9767 | 0.6446 | 16.29 |

These timings were measured in prior configurations and must not be conflated with HTTP end-to-end latency or freshly measured runner hardware performance.

## Matrix F — Archived concurrency sweep

| Concurrency | Queries | p50 (ms) | p95 (ms) | p99 (ms) | RPS | Errors |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 3.98 | 4.41 | 5.03 | 244.3 | 0 |
| 2 | 100 | 5.53 | 6.30 | 6.45 | 353.6 | 0 |
| 5 | 100 | 11.49 | 13.75 | 14.61 | 421.3 | 0 |
| 10 | 100 | 21.93 | 31.51 | 36.57 | 415.9 | 0 |
| 20 | 100 | 36.56 | 63.37 | 70.96 | 401.3 | 0 |

### SciFact runtime tail latency (archived)

| Metric | Fixed Hybrid | Adaptive Router |
|---|---:|---:|
| Total Latency p50 (ms) | 43.74 ms | 34.00 ms |
| Total Latency p95 (ms) | 72.31 ms | 70.69 ms |
| Total Latency p99 (ms) | 88.55 ms | 101.23 ms |

SciFact adaptive routing historically avoided 138/300 dense calls, yet had WORSE p99 tail latency (101.23 vs 88.55 ms). Do not claim universal latency improvement.

## Matrix G — Generation and grounding (noncomparable populations)

| Generator | Mode | Cases | Token F1 | Faithfulness | Citation support | Abstention | Mean generation ms |
|---|---|---:|---:|---:|---:|---:|---:|
| mock-generator | mock | 20 | 0.3243 | 0.6250 | 1.0000 | 1.0000 | 0.01 |
| qwen2.5:0.5b | neural | 10 | 0.3791 | N/A | N/A | 1.0000 | 11443.84 |

Do NOT pool the mock 20-case and neural 10-case results. Neural-level citation support and faithfulness have not been established in this summary. Do not infer generalizable security/safety effectiveness from 10 cases.

## Findings and negatives

- On historical FiQA results, Dense Exact outperforms Fixed Hybrid (0.3532 vs 0.3300 nDCG@10, stored bootstrap p=0.0296). Hybrid is not universally better.
- On NFCorpus, Fixed Hybrid outperforms Adaptive Router in archived nDCG@10; router accuracy is domain-dependent.
- On SciFact, adaptive routing avoids expensive dense calls, but Stage 2 hybrid headline has an unresolved registry-vs-bootstrap inconsistency.

## Reconciliation / limitations

- **UNRESOLVED** SciFact D50 Fixed Hybrid nDCG@10: registry=0.6851, matched bootstrap=0.6799; unresolved until source rankings are independently re-evaluated.
- **UNRESOLVED** SciFact D50 Fixed Hybrid Recall@10: registry=0.8254, matched bootstrap=0.8104; unresolved until source rankings are independently re-evaluated.
- The former README top-line values disagree with the canonical CSV/paired statistics. The revised README should use clearly labeled archived values, not mixed pipeline depths.
- The full BEIR dataset, cached dense embeddings and physical Qdrant engine are NOT committed; external data retrieval is required to independently rerun retrieval scores, statistical tests and hardware latency.
- Historic load-test results are not CI remeasurements. No scientific claim here is based on freshly generated neural answers.
- Verified from saved records: row counts, score ranges, matched paired comparisons, generation case counts and document/query inventories. This does not verify original data provenance.

## Reproduce artifact-level report (offline)

Run: python scripts/build_evaluation_matrix_report.py
Verify: python scripts/build_evaluation_matrix_report.py --check

Machine-readable companion: results/evaluation_matrix_report.json.
