# Full BEIR Reevaluation

Execution date: 2026-10-24 (Asia/Kolkata). Repository baseline: `1f9dad3`. The evaluation-matrix report reviewed came from remote feature commit `3137b2212da27eab40f26f0edad31a2a9daf0def`, not main; its exact text is retained in [the archived input](../results/recomputed/ARCHIVED_EVALUATION_MATRIX_REPORT.md).

## Verdict and evidence boundaries

**Retrieval reproduction and SciFact reconciliation: PASS. Full end-to-end RAG scientific evaluation: BLOCKED.** No new neural answers or validated claim-level judgments are claimed. Historical registry integrity and the default Turbopack build have failures documented below; an alternate production webpack build passed.

This report separates (1) newly encoded datasets and newly executed retrieval/server experiments, (2) recomputation/replay from saved ranked outputs, (3) archived summaries, and (4) mock fixtures and unverified claims. “PASS” means the specified checks completed, not that a system outperforms another or is production-ready.

## Official dataset verification and leakage

Archives were freshly acquired over verified HTTPS from the official BEIR distribution, matched against BEIR's published MD5 values, and compared with the local archives. SHA-256 digests were independently calculated, matched between downloaded/local bytes, and pinned in the acquisition script; they are **not upstream-published SHA-256 values**. Every extracted corpus/query/qrels file was byte-checked against its archive. IDs are unique, qrel references resolve, and no benchmark labels were generated or substituted.

Sources: [BEIR dataset catalog](https://github.com/beir-cellar/beir), [SciFact card](https://huggingface.co/datasets/BeIR/scifact), [NFCorpus card](https://huggingface.co/datasets/BeIR/nfcorpus), [FiQA card](https://huggingface.co/datasets/BeIR/fiqa). All three BEIR cards declare CC BY-SA 4.0; this records their declarations, not a legal audit of underlying source material. See [DATA_LICENSES.md](../DATA_LICENSES.md).

| Dataset | Corpus docs | All query records | Train / dev / test query counts | Test qrel rows | Test grades |
|---|---|---|---|---|---|
| scifact | 5183 | 1109 | 809 / absent / 300 | 339 | [1] |
| nfcorpus | 3633 | 3237 | 2590 / 324 / 323 | 12334 | [1, 2] |
| fiqa | 57638 | 6648 | 5500 / 500 / 648 | 1706 | [1] |

| Dataset | Archive SHA-256 |
|---|---|
| scifact | 536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165 |
| nfcorpus | efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b |
| fiqa | 32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02 |

| Dataset | Split pair | Shared query IDs | Shared normalized exact query texts |
|---|---|---|---|
| scifact | test__train | 0 | 2 |
| nfcorpus | dev__test | 0 | 1 |
| nfcorpus | dev__train | 0 | 4 |
| nfcorpus | test__train | 0 | 4 |
| fiqa | dev__test | 0 | 0 |
| fiqa | dev__train | 0 | 0 |
| fiqa | test__train | 0 | 0 |

The shared corpus is intentional in retrieval; document overlap across qrels is not itself query leakage. Exact normalized query text overlap is a real limitation even when IDs differ. SciFact train/test shares two texts; NFCorpus dev/test also shares a text. Calibration replay excludes matching test texts in a separate ablation, without removing test queries. Exact-text checks do not rule out paraphrases, model-pretraining contamination, or undisclosed historical test-set tuning. Full overlap strings and component hashes are in [dataset_verification.json](../results/recomputed/dataset_verification.json).

## Retrieval protocol and metric definitions

All four systems use exactly each dataset's official test-qrels query population: 300 / 323 / 648 queries. Documents are `title + space + text`, with no synthetic corpus expansion or chunk-level substitute labels. Dense embeddings are newly generated with `sentence-transformers/all-MiniLM-L6-v2`, pinned revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, CPU float32, 384 dimensions, L2 normalization, maximum sequence length 256. Document/query batch sizes are 64/32. Model truncation can discard trailing document content.

BM25 is the repository's regex tokenizer and smoothed RSJ variant (`k1=1.5`, `b=0.75`), not an Elasticsearch baseline. Dense exact search exhaustively scores the complete corpus. Equal dense scores use stable corpus input order; the legacy reverse-argsort tie order is separately audited. RRF uses `k=60`. Candidate depths **50, 100, 250 per branch** are explicitly matched; returned depth equals candidate depth. Quality is evaluated at the same top-10 cutoff. The router's main matrix uses fixed defaults (margin 0.28, IDF 6, dense-confidence 0.75, weak-BM25 8), with no development/test fitting. Historical calibration and feature ablations are separate below.

Recall@k counts unique relevant documents / all positive qrels; Precision@k divides by k even when fewer documents are returned. Graded nDCG@10 uses linear judgment gain and log2(rank+1) discount; binary nDCG is retained only as a legacy diagnostic. MRR@10 is the reciprocal first relevant rank, censored at 10. Full retained-depth MRR is separately named MRR@50/100/250 in the raw metric JSON. Unjudged documents have gain zero. Query IDs are not semantically excluded from document IDs; this follows the repository's protocol, and differs from evaluator configurations that automatically exclude identical IDs.

## Fresh retrieval quality: dataset / model / configuration matrix

The model column distinguishes lexical-only BM25 from MiniLM dense retrieval; Hybrid and Router combine BM25 with that same encoder. These results do not establish performance for other embedding models. All values below are newly executed; archived numbers are not inserted into missing cells.

| Dataset | System | Candidates / returned | R@1 | R@5 | R@10 | graded nDCG@10 | MRR@10 | P@1 | P@5 | P@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| scifact | BM25 | 50 | 0.515000 | 0.702778 | 0.760500 | 0.643227 | 0.609525 | 0.526667 | 0.150667 | 0.083667 |
| scifact | Dense_Exact | 50 | 0.482333 | 0.737944 | 0.783333 | 0.645082 | 0.604725 | 0.503333 | 0.164000 | 0.088333 |
| scifact | Fixed_Hybrid | 50 | 0.529833 | 0.737556 | 0.825389 | 0.685139 | 0.645169 | 0.546667 | 0.161333 | 0.092333 |
| scifact | Adaptive_Router | 50 | 0.539833 | 0.739222 | 0.823722 | 0.688230 | 0.650364 | 0.556667 | 0.161333 | 0.092000 |
| scifact | BM25 | 100 | 0.515000 | 0.702778 | 0.760500 | 0.643227 | 0.609525 | 0.526667 | 0.150667 | 0.083667 |
| scifact | Dense_Exact | 100 | 0.482333 | 0.737944 | 0.783333 | 0.645082 | 0.604725 | 0.503333 | 0.164000 | 0.088333 |
| scifact | Fixed_Hybrid | 100 | 0.529833 | 0.729222 | 0.800389 | 0.676341 | 0.641123 | 0.546667 | 0.160000 | 0.090000 |
| scifact | Adaptive_Router | 100 | 0.539833 | 0.730889 | 0.802056 | 0.680618 | 0.646873 | 0.556667 | 0.160000 | 0.090000 |
| scifact | BM25 | 250 | 0.515000 | 0.702778 | 0.760500 | 0.643227 | 0.609525 | 0.526667 | 0.150667 | 0.083667 |
| scifact | Dense_Exact | 250 | 0.482333 | 0.737944 | 0.783333 | 0.645082 | 0.604725 | 0.503333 | 0.164000 | 0.088333 |
| scifact | Fixed_Hybrid | 250 | 0.529833 | 0.732556 | 0.810389 | 0.679874 | 0.642783 | 0.546667 | 0.160667 | 0.091000 |
| scifact | Adaptive_Router | 250 | 0.539833 | 0.734222 | 0.808722 | 0.683188 | 0.648200 | 0.556667 | 0.160667 | 0.090667 |
| nfcorpus | BM25 | 50 | 0.047534 | 0.108344 | 0.139647 | 0.293725 | 0.504610 | 0.427245 | 0.274923 | 0.212074 |
| nfcorpus | Dense_Exact | 50 | 0.043897 | 0.119653 | 0.154988 | 0.316673 | 0.507654 | 0.421053 | 0.303406 | 0.243344 |
| nfcorpus | Fixed_Hybrid | 50 | 0.051678 | 0.125480 | 0.159291 | 0.329260 | 0.540503 | 0.458204 | 0.309598 | 0.243963 |
| nfcorpus | Adaptive_Router | 50 | 0.052297 | 0.124071 | 0.157569 | 0.326038 | 0.541953 | 0.461300 | 0.303406 | 0.239628 |
| nfcorpus | BM25 | 100 | 0.047534 | 0.108344 | 0.139647 | 0.293725 | 0.504610 | 0.427245 | 0.274923 | 0.212074 |
| nfcorpus | Dense_Exact | 100 | 0.043897 | 0.119653 | 0.154988 | 0.316673 | 0.507654 | 0.421053 | 0.303406 | 0.243344 |
| nfcorpus | Fixed_Hybrid | 100 | 0.051678 | 0.124694 | 0.158931 | 0.330444 | 0.538448 | 0.458204 | 0.310836 | 0.244892 |
| nfcorpus | Adaptive_Router | 100 | 0.052297 | 0.123285 | 0.157951 | 0.327941 | 0.540414 | 0.461300 | 0.304644 | 0.241486 |
| nfcorpus | BM25 | 250 | 0.047534 | 0.108344 | 0.139647 | 0.293725 | 0.504610 | 0.427245 | 0.274923 | 0.212074 |
| nfcorpus | Dense_Exact | 250 | 0.043897 | 0.119653 | 0.154988 | 0.316673 | 0.507654 | 0.421053 | 0.303406 | 0.243344 |
| nfcorpus | Fixed_Hybrid | 250 | 0.051678 | 0.125241 | 0.158444 | 0.330088 | 0.537923 | 0.458204 | 0.312074 | 0.243653 |
| nfcorpus | Adaptive_Router | 250 | 0.052297 | 0.123778 | 0.157645 | 0.327470 | 0.539889 | 0.461300 | 0.305263 | 0.240248 |
| fiqa | BM25 | 50 | 0.101348 | 0.224405 | 0.284253 | 0.224675 | 0.281249 | 0.211420 | 0.096605 | 0.063735 |
| fiqa | Dense_Exact | 50 | 0.171403 | 0.367056 | 0.441306 | 0.368671 | 0.445122 | 0.347222 | 0.167593 | 0.104475 |
| fiqa | Fixed_Hybrid | 50 | 0.152769 | 0.335926 | 0.429221 | 0.342915 | 0.413618 | 0.313272 | 0.148457 | 0.097840 |
| fiqa | Adaptive_Router | 50 | 0.153540 | 0.334640 | 0.425363 | 0.341710 | 0.414014 | 0.314815 | 0.148148 | 0.097377 |
| fiqa | BM25 | 100 | 0.101348 | 0.224405 | 0.284253 | 0.224675 | 0.281249 | 0.211420 | 0.096605 | 0.063735 |
| fiqa | Dense_Exact | 100 | 0.171403 | 0.367056 | 0.441306 | 0.368671 | 0.445122 | 0.347222 | 0.167593 | 0.104475 |
| fiqa | Fixed_Hybrid | 100 | 0.152769 | 0.325285 | 0.421807 | 0.338374 | 0.410416 | 0.313272 | 0.144753 | 0.095988 |
| fiqa | Adaptive_Router | 100 | 0.153540 | 0.323999 | 0.417949 | 0.337196 | 0.410860 | 0.314815 | 0.144444 | 0.095525 |
| fiqa | BM25 | 250 | 0.101348 | 0.224405 | 0.284253 | 0.224675 | 0.281249 | 0.211420 | 0.096605 | 0.063735 |
| fiqa | Dense_Exact | 250 | 0.171403 | 0.367056 | 0.441306 | 0.368671 | 0.445122 | 0.347222 | 0.167593 | 0.104475 |
| fiqa | Fixed_Hybrid | 250 | 0.152769 | 0.331715 | 0.419739 | 0.338359 | 0.411223 | 0.313272 | 0.147840 | 0.095679 |
| fiqa | Adaptive_Router | 250 | 0.153540 | 0.330429 | 0.415881 | 0.337182 | 0.411667 | 0.314815 | 0.147531 | 0.095216 |

## SciFact D50 discrepancy: root cause and correction

The old `audit_and_verify_phase4.py` saved only **50 BM25 and 50 dense hits** for each query, although its stored fused hybrid list came from depth 100. The old registry builder then re-fused those saved top-50 branches, called the result the Stage 2 `top_k=50` run, and described 250 candidates without having the missing 200 candidates. The actual Stage 2 policy requests `max(top_k * 5, 50)`: returning 50 therefore means **250 candidates per branch**. The label “D50” conflated returned cutoff with branch depth. Changing the returned cutoff alone cannot change a fixed fused ranking's first ten documents.

We reloaded official qrels, verified all 300 query IDs, recomputed the saved branch outputs, rebuilt branch lists to 250 using archived vectors, and independently repeated retrieval with fresh embeddings. The reconstructed top-50 branches match the archive for every query. Both disputed values are reproducible under their correct configurations:

| Evidence / configuration | nDCG@10 | Recall@10 | MRR@10 | MRR@50 |
|---|---|---|---|---|
| stored_Hybrid_Exact_RRF_D100 | 0.676341 | 0.800389 | 0.641123 | 0.646708 |
| refused_stored_branches_D50 | 0.685139 | 0.825389 | 0.645169 | 0.650141 |
| reconstructed_branch_D50 | 0.685139 | 0.825389 | 0.645169 | 0.650141 |
| reconstructed_branch_D100 | 0.676341 | 0.800389 | 0.641123 | 0.646708 |
| reconstructed_branch_D250 | 0.679874 | 0.810389 | 0.642783 | 0.647971 |

The corrected `EXP-SCIFACT-HYBRID-D50` entry means **250 branch candidates, 50 returned**, yielding 0.6798740045 nDCG@10 and 0.8103888889 Recall@10. Its MRR@10 is 0.6427830688; the former mixed-depth MRR 0.6452 must not be carried forward. The 0.6851393844 / 0.8253888889 result remains a valid **50-candidate** ablation, not the matched Stage 2 run.

The fix changes the data path, not just the display: `FixedHybridPolicy` and `AdaptiveRouterPolicy` now accept explicit candidate depth; regression tests verify invariant top-10 prefixes across returned cutoffs at fixed depth. The registry builder recomputes quality from ranked documents and qrels, checks population completeness and aggregate consistency, and removes hard-coded measurements and conclusions. The original registry is preserved as [an archived input](../results/recomputed/archived_canonical_results_registry.json). [Reconciliation evidence](../results/recomputed/scifact_reconciliation/summary.json) includes hashes and [ranked outputs](../results/recomputed/scifact_reconciliation/per_query.jsonl.gz).

## Other inconsistencies and calibrated-router replication

NFCorpus has relevance grades 1 and 2; the earlier scripts converted them into sets, reporting binary nDCG as nDCG without qualification. The new main matrix preserves grades. The legacy NFCorpus Dense Exact binary nDCG 0.3181067291 is reproducible with its reverse-argsort tie ordering; stable tie ordering gives 0.3184208417, while graded stable nDCG is 0.3166731275. The drift audit isolates tie ordering from tiny query-embedding floating-point differences. Newly rebuilt SciFact and NFCorpus corpus vectors exactly match the archived corpus vectors.

The original FiQA calibration function slices `dev_qids[:300]`, despite a prior report saying all 500 development queries were used. NFCorpus's actual calibration selects margin 0.15 / IDF 4.0, not the hard-coded registry description 0.25 / 6.0. Calibration is reproduced directly from the original function bodies, with an additional exact-text-deduplicated variant. These are exploratory replications, not preregistered selection or equivalence trials.

| Dataset | 250-candidate router | graded nDCG@10 | binary nDCG@10 | Recall@10 | Lexical routes (replay) |
|---|---|---|---|---|---|
| scifact | Router_deduplicated_calibration | 0.686123 | 0.686123 | 0.809278 | 138/300 |
| scifact | Router_fixed_defaults | 0.683188 | 0.683188 | 0.808722 | 88/300 |
| scifact | Router_historical | 0.686123 | 0.686123 | 0.809278 | 138/300 |
| nfcorpus | Router_deduplicated_calibration | 0.321360 | 0.321135 | 0.154549 | 95/323 |
| nfcorpus | Router_fixed_defaults | 0.327470 | 0.328374 | 0.157645 | 39/323 |
| nfcorpus | Router_historical | 0.321360 | 0.321135 | 0.154549 | 95/323 |
| fiqa | Router_deduplicated_calibration | 0.329346 | 0.329346 | 0.406468 | 66/648 |
| fiqa | Router_fixed_defaults | 0.337182 | 0.337182 | 0.415881 | 21/648 |
| fiqa | Router_historical | 0.329346 | 0.329346 | 0.406468 | 66/648 |

Historical SciFact calibration reproduces 0.6861231432 nDCG@10 and 138/300 lexical-only routes. The main fixed-default matrix is a different router configuration and must not be silently substituted for it. Replay lexical routes describe search decisions; all query embeddings had already been computed, so these counts do not measure avoided neural computation in the fresh run.

## Paired inference

Statistics use query-aligned paired differences, 10,000 bootstrap resamples, seed 42, percentile 95% confidence intervals, and a two-sided null-centered bootstrap p-value with a +1 finite-resample correction. All test queries are required on both sides. Holm (FWER) and Benjamini-Hochberg (FDR) adjustments cover **36 comparisons per dataset** in the main matrix: six system pairs × three depths × two metrics. Calibrated-router replay has its own 12-comparison family; ANN has a separate family of all exact/ANN configuration pairs × two metrics. Corrections are not global across those families or datasets. Tests are descriptive/exploratory. No non-inferiority or equivalence claim is made; a nonsignificant result is not proof of equality.

Selected matched 250-candidate main comparisons follow. Complete results for both nDCG@10 and Recall@10 are in each dataset's `statistics.json`.

| Dataset | Pair | Delta | 95% CI | Raw p | Holm p | BH q |
|---|---|---|---|---|---|---|
| scifact | Adaptive_Router - BM25 | 0.039961 | [0.018639, 0.060648] | 0.000100 | 0.003600 | 0.001200 |
| scifact | Adaptive_Router - Dense_Exact | 0.038106 | [0.007430, 0.067888] | 0.013999 | 0.308000 | 0.033600 |
| scifact | Adaptive_Router - Fixed_Hybrid | 0.003314 | [-0.003265, 0.010961] | 0.360164 | 1.000000 | 0.463100 |
| scifact | BM25 - Dense_Exact | -0.001855 | [-0.041571, 0.038506] | 0.933307 | 1.000000 | 0.933300 |
| scifact | BM25 - Fixed_Hybrid | -0.036647 | [-0.058883, -0.013972] | 0.000900 | 0.027000 | 0.004600 |
| scifact | Dense_Exact - Fixed_Hybrid | -0.034792 | [-0.063359, -0.005128] | 0.018498 | 0.388500 | 0.041600 |
| nfcorpus | Adaptive_Router - BM25 | 0.033745 | [0.021736, 0.046226] | 0.000100 | 0.003600 | 0.000600 |
| nfcorpus | Adaptive_Router - Dense_Exact | 0.010797 | [-0.007274, 0.028753] | 0.240876 | 1.000000 | 0.361300 |
| nfcorpus | Adaptive_Router - Fixed_Hybrid | -0.002618 | [-0.008621, 0.002777] | 0.362864 | 1.000000 | 0.483800 |
| nfcorpus | BM25 - Dense_Exact | -0.022949 | [-0.046205, -0.000168] | 0.050095 | 1.000000 | 0.120200 |
| nfcorpus | BM25 - Fixed_Hybrid | -0.036364 | [-0.049744, -0.023249] | 0.000100 | 0.003600 | 0.000600 |
| nfcorpus | Dense_Exact - Fixed_Hybrid | -0.013415 | [-0.030106, 0.003247] | 0.115288 | 1.000000 | 0.207500 |
| fiqa | Adaptive_Router - BM25 | 0.112506 | [0.096689, 0.128382] | 0.000100 | 0.003600 | 0.000200 |
| fiqa | Adaptive_Router - Dense_Exact | -0.031490 | [-0.052259, -0.011042] | 0.002600 | 0.044200 | 0.004600 |
| fiqa | Adaptive_Router - Fixed_Hybrid | -0.001178 | [-0.003659, 0.001088] | 0.313769 | 1.000000 | 0.313800 |
| fiqa | BM25 - Dense_Exact | -0.143996 | [-0.170967, -0.117737] | 0.000100 | 0.003600 | 0.000200 |
| fiqa | BM25 - Fixed_Hybrid | -0.113684 | [-0.129680, -0.097744] | 0.000100 | 0.003600 | 0.000200 |
| fiqa | Dense_Exact - Fixed_Hybrid | 0.030312 | [0.010117, 0.050972] | 0.003200 | 0.048000 | 0.005200 |

## Fresh search timing, memory and index build measurements

Search timing is one serial pass per configuration, using **precomputed query vectors**. It excludes query encoding, generation and network for the NumPy/BM25 matrix. QPS is inverse mean measured search time, not load-tested service capacity. Qdrant timing includes local HTTP overhead, with a warm-up query. Runs shared this laptop with other audit work (including embedding/build processes); these are observational timings, **not isolated performance comparisons**. No thermal, energy, GPU, concurrency-capacity or latency-improvement claim is made. p99 is based on only 300/323/648 queries and has limited tail precision.

RSS values are process-wide snapshots or process peak working set, not per-index incremental memory. Qdrant RSS includes collections retained from previous configurations. Index sizes are actual serialized lexical index, float32 array file, or full Qdrant collection-directory bytes (including storage overhead/WAL), not theoretical quantization ratios.

| Dataset | Corpus encoding s | Query encoding s | BM25 build s | Dense array bytes | BM25 index bytes | Process peak working set bytes |
|---|---|---|---|---|---|---|
| scifact | 310.226130 | 2.377700 | 0.955890 | 7961216 | 5953114 | 1030492160 |
| nfcorpus | 276.964359 | 0.917976 | 0.914138 | 5580416 | 4305682 | 1029894144 |
| fiqa | 2270.033207 | 2.879952 | 9.715339 | 88532096 | 36395927 | 1432141824 |

| Dataset | System | Depth | p50 ms | p95 ms | p99 ms | Serial search QPS | Process RSS bytes |
|---|---|---|---|---|---|---|---|
| scifact | BM25 | 50 | 12.141600 | 27.889410 | 35.674107 | 74.946610 | 699711488 |
| scifact | Dense_Exact | 50 | 0.996300 | 1.386170 | 1.519312 | 950.979541 | 699711488 |
| scifact | Fixed_Hybrid | 50 | 13.390350 | 29.065820 | 35.072643 | 69.078911 | 699711488 |
| scifact | Adaptive_Router | 50 | 12.999700 | 29.946650 | 36.083538 | 69.396973 | 699711488 |
| scifact | BM25 | 100 | 12.285600 | 29.955875 | 39.762769 | 71.238507 | 713216000 |
| scifact | Dense_Exact | 100 | 1.038350 | 1.504950 | 1.874355 | 902.902742 | 713216000 |
| scifact | Fixed_Hybrid | 100 | 13.535900 | 30.270595 | 53.994077 | 64.903931 | 713216000 |
| scifact | Adaptive_Router | 100 | 13.094600 | 29.440425 | 48.254730 | 67.383742 | 713220096 |
| scifact | BM25 | 250 | 13.354050 | 33.204560 | 70.363533 | 63.661791 | 745611264 |
| scifact | Dense_Exact | 250 | 1.164950 | 1.678125 | 2.577099 | 789.340535 | 745611264 |
| scifact | Fixed_Hybrid | 250 | 14.963550 | 36.807325 | 55.266821 | 57.389668 | 745611264 |
| scifact | Adaptive_Router | 250 | 14.457050 | 36.219125 | 52.132942 | 59.006394 | 745611264 |
| nfcorpus | BM25 | 50 | 0.440000 | 6.897450 | 9.869310 | 499.551100 | 778469376 |
| nfcorpus | Dense_Exact | 50 | 0.696500 | 1.013700 | 1.112242 | 1384.267521 | 778469376 |
| nfcorpus | Fixed_Hybrid | 50 | 1.228900 | 8.089310 | 11.147406 | 349.708610 | 778469376 |
| nfcorpus | Adaptive_Router | 50 | 1.243600 | 8.154790 | 10.770826 | 360.739241 | 778469376 |
| nfcorpus | BM25 | 100 | 0.421300 | 6.561320 | 8.861634 | 539.231693 | 792014848 |
| nfcorpus | Dense_Exact | 100 | 0.700100 | 0.973650 | 1.057650 | 1395.670397 | 792014848 |
| nfcorpus | Fixed_Hybrid | 100 | 1.317700 | 7.765790 | 9.830156 | 366.195174 | 792023040 |
| nfcorpus | Adaptive_Router | 100 | 1.244600 | 7.869030 | 10.042228 | 368.573611 | 792023040 |
| nfcorpus | BM25 | 250 | 0.398500 | 6.616790 | 8.563540 | 547.326918 | 823734272 |
| nfcorpus | Dense_Exact | 250 | 0.811200 | 1.170810 | 1.308816 | 1193.016385 | 823734272 |
| nfcorpus | Fixed_Hybrid | 250 | 1.529600 | 7.877760 | 9.319044 | 346.279171 | 823734272 |
| nfcorpus | Adaptive_Router | 250 | 1.521300 | 8.014800 | 9.855986 | 359.482554 | 823734272 |
| fiqa | BM25 | 50 | 139.692200 | 275.327910 | 316.448018 | 7.010837 | 1064071168 |
| fiqa | Dense_Exact | 50 | 8.391650 | 9.487980 | 11.729314 | 116.403107 | 1064071168 |
| fiqa | Fixed_Hybrid | 50 | 148.334150 | 280.117920 | 326.629178 | 6.589253 | 1064071168 |
| fiqa | Adaptive_Router | 50 | 148.797550 | 279.352245 | 329.538898 | 6.653771 | 1064071168 |
| fiqa | BM25 | 100 | 139.118850 | 267.955875 | 319.952535 | 7.070220 | 1094606848 |
| fiqa | Dense_Exact | 100 | 8.407950 | 9.397305 | 10.529787 | 116.940516 | 1094606848 |
| fiqa | Fixed_Hybrid | 100 | 147.095350 | 281.561650 | 341.046957 | 6.661523 | 1094606848 |
| fiqa | Adaptive_Router | 100 | 146.938000 | 282.065640 | 337.548473 | 6.632850 | 1094606848 |
| fiqa | BM25 | 250 | 141.872550 | 266.037580 | 332.734701 | 7.016148 | 1166270464 |
| fiqa | Dense_Exact | 250 | 8.505800 | 9.561125 | 11.570349 | 114.803547 | 1166270464 |
| fiqa | Fixed_Hybrid | 250 | 149.610150 | 276.076685 | 338.957632 | 6.574609 | 1166270464 |
| fiqa | Adaptive_Router | 250 | 149.617800 | 281.368115 | 352.099639 | 6.590590 | 1166270464 |

## Real Qdrant HNSW and quantization

Server: Qdrant 1.19.2, isolated HTTP port 6335, CPU indexing. Each collection is required to be GREEN with its **full corpus indexed** before evaluation. No in-memory client fallback is used. Standard HNSW uses M=16 / ef_construct=100; tuned and quantized variants use M=32 / ef_construct=200. HNSW ef is 32 or 128; INT8 uses quantile 0.99; binary uses 1-bit quantization. Quantization rescore is ablated at ef=128, oversampling=2. Index/full-scan thresholds are 10 KB (the server rejected the first attempt's value 1). HNSW construction has no exposed fixed random seed here; exact reruns can differ. The [Qdrant quantization documentation](https://qdrant.tech/documentation/manage-data/quantization/) describes the approximation/rescoring tradeoff.

ANN Recall@10 is intersection with freshly computed exact dense top-10 / 10, distinct from relevance-judgment Recall@10. Equal-distance neighbors can affect exact-set overlap. Rescoring cannot recover neighbors absent from the retrieved candidate pool. These settings are not assumed identical to archived ANN experiments.

| Dataset | Configuration | ANN R@10 | Gold R@10 | graded nDCG@10 | p50 ms | p95 ms | p99 ms | Serial HTTP QPS | Build/upload/optimize s | Collection bytes | Server RSS bytes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| scifact | hnsw_ef32_rescoreFalse | 0.975667 | 0.770000 | 0.634936 | 7.865350 | 26.876670 | 43.310897 | 90.370978 | 4.348189 | 145748321 | 96747520 |
| scifact | hnsw_ef128_rescoreFalse | 0.997667 | 0.780000 | 0.641748 | 8.528250 | 31.499730 | 44.964443 | 83.571112 | 4.348189 | 145748321 | 96788480 |
| scifact | hnsw_tuned_ef32_rescoreFalse | 0.984667 | 0.783333 | 0.645809 | 8.234400 | 16.763855 | 29.540824 | 108.549256 | 6.278409 | 145788153 | 104828928 |
| scifact | hnsw_tuned_ef128_rescoreFalse | 0.999333 | 0.783333 | 0.645082 | 10.789700 | 32.876595 | 65.892980 | 66.918618 | 6.278409 | 145788153 | 105066496 |
| scifact | int8_ef128_rescoreFalse | 0.981333 | 0.785000 | 0.646451 | 9.918450 | 35.548130 | 54.116065 | 69.278353 | 6.593257 | 147799422 | 117452800 |
| scifact | int8_ef128_rescoreTrue | 0.998667 | 0.783333 | 0.645082 | 7.776500 | 23.899300 | 31.128376 | 98.147258 | 6.593257 | 147799422 | 117583872 |
| scifact | binary_ef128_rescoreFalse | 0.561667 | 0.711833 | 0.585390 | 8.337500 | 30.238900 | 58.321773 | 84.280650 | 4.568863 | 146082856 | 127561728 |
| scifact | binary_ef128_rescoreTrue | 0.738333 | 0.748333 | 0.630193 | 9.446850 | 36.541415 | 54.890160 | 67.660505 | 4.568863 | 146082856 | 127827968 |
| nfcorpus | hnsw_ef32_rescoreFalse | 0.921672 | 0.142649 | 0.304702 | 8.885300 | 28.059080 | 33.795462 | 83.429490 | 3.596881 | 143246967 | 135815168 |
| nfcorpus | hnsw_ef128_rescoreFalse | 0.989783 | 0.150220 | 0.311447 | 8.102400 | 25.734860 | 33.895780 | 95.823081 | 3.596881 | 143246967 | 136318976 |
| nfcorpus | hnsw_tuned_ef32_rescoreFalse | 0.946130 | 0.151864 | 0.312905 | 7.738100 | 24.643000 | 31.239110 | 96.955976 | 3.373058 | 143266936 | 144814080 |
| nfcorpus | hnsw_tuned_ef128_rescoreFalse | 0.995666 | 0.153440 | 0.314584 | 7.388200 | 27.590330 | 31.923688 | 95.162431 | 3.373058 | 143266936 | 145018880 |
| nfcorpus | int8_ef128_rescoreFalse | 0.983591 | 0.153134 | 0.313823 | 7.459000 | 26.323550 | 30.871976 | 100.760715 | 3.707264 | 144677425 | 153268224 |
| nfcorpus | int8_ef128_rescoreTrue | 0.996594 | 0.153440 | 0.314924 | 7.847700 | 26.993690 | 32.272434 | 89.025671 | 3.707264 | 144677425 | 153366528 |
| nfcorpus | binary_ef128_rescoreFalse | 0.526625 | 0.131938 | 0.275749 | 7.952400 | 27.535550 | 32.375200 | 87.622916 | 3.510611 | 143471953 | 159625216 |
| nfcorpus | binary_ef128_rescoreTrue | 0.697214 | 0.142707 | 0.301089 | 8.446300 | 27.750800 | 38.072140 | 88.304124 | 3.510611 | 143471953 | 159727616 |
| fiqa | hnsw_ef32_rescoreFalse | 0.959568 | 0.425205 | 0.356481 | 6.147400 | 19.815270 | 26.935532 | 131.155597 | 32.934214 | 264277278 | 258297856 |
| fiqa | hnsw_ef128_rescoreFalse | 0.995216 | 0.437190 | 0.365037 | 6.958700 | 20.498805 | 27.967927 | 118.036962 | 32.934214 | 264277278 | 258301952 |
| fiqa | hnsw_tuned_ef32_rescoreFalse | 0.977623 | 0.435364 | 0.363157 | 6.669800 | 22.560280 | 28.058936 | 119.013410 | 55.078400 | 265343258 | 357228544 |
| fiqa | hnsw_tuned_ef128_rescoreFalse | 0.997994 | 0.439762 | 0.367134 | 7.799450 | 17.380510 | 28.844031 | 110.609169 | 55.078400 | 265343258 | 357392384 |
| fiqa | int8_ef128_rescoreFalse | 0.981173 | 0.443276 | 0.367948 | 7.257450 | 21.894555 | 27.715240 | 105.939228 | 32.197367 | 287711668 | 477437952 |
| fiqa | int8_ef128_rescoreTrue | 0.998302 | 0.441820 | 0.368887 | 7.071300 | 22.259890 | 28.366772 | 112.549277 | 32.197367 | 287711668 | 477503488 |
| fiqa | binary_ef128_rescoreFalse | 0.544290 | 0.368684 | 0.301310 | 6.810000 | 18.767870 | 27.854016 | 116.747019 | 23.521796 | 268993693 | 583704576 |
| fiqa | binary_ef128_rescoreTrue | 0.714660 | 0.408240 | 0.346104 | 6.834950 | 23.124550 | 27.969985 | 116.090718 | 23.521796 | 268993693 | 578646016 |

Other ANN quality metrics (Recall@1/5, Precision@1/5/10 and MRR@10), paired statistics, per-query ranks, and actual collection configurations are retained under each dataset's `qdrant/` directory. Binary quantization's substantial ANN distortion under these settings is a negative result, not evidence of universal compression benefits.

## Fusion and router ablations

These are recomputed from the **new** BM25/dense candidate lists, not new timing measurements. Candidate-depth changes appear in the main matrix; the matrix below additionally varies RRF constant and removes margin/IDF gates or forces hybrid fallback. Fixed-default feature ablations do not inherit historically calibrated thresholds. No significance is inferred merely from these point differences.

| Dataset | Depth | Ablation | graded nDCG@10 | Recall@10 | MRR@10 | Lexical routes |
|---|---|---|---|---|---|---|
| scifact | 50 | RRF_k10 | 0.694525 | 0.837222 | 0.652440 | 0 |
| scifact | 50 | RRF_k100 | 0.684900 | 0.825389 | 0.644910 | 0 |
| scifact | 50 | RRF_k30 | 0.686074 | 0.826222 | 0.645626 | 0 |
| scifact | 50 | RRF_k60 | 0.685139 | 0.825389 | 0.645169 | 0 |
| scifact | 50 | Router_always_hybrid | 0.685139 | 0.825389 | 0.645169 | 0 |
| scifact | 50 | Router_deduplicated_calibration | 0.691022 | 0.824278 | 0.654082 | 138 |
| scifact | 50 | Router_fixed_defaults | 0.688230 | 0.823722 | 0.650364 | 88 |
| scifact | 50 | Router_historical | 0.691022 | 0.824278 | 0.654082 | 138 |
| scifact | 50 | Router_no_idf | 0.688230 | 0.823722 | 0.650364 | 97 |
| scifact | 50 | Router_no_margin | 0.657443 | 0.778500 | 0.623307 | 245 |
| scifact | 250 | RRF_k10 | 0.694249 | 0.832222 | 0.653329 | 0 |
| scifact | 250 | RRF_k100 | 0.677255 | 0.803722 | 0.641321 | 0 |
| scifact | 250 | RRF_k30 | 0.679495 | 0.803889 | 0.643587 | 0 |
| scifact | 250 | RRF_k60 | 0.679874 | 0.810389 | 0.642783 | 0 |
| scifact | 250 | Router_always_hybrid | 0.679874 | 0.810389 | 0.642783 | 0 |
| scifact | 250 | Router_deduplicated_calibration | 0.686123 | 0.809278 | 0.652112 | 138 |
| scifact | 250 | Router_fixed_defaults | 0.683188 | 0.808722 | 0.648200 | 88 |
| scifact | 250 | Router_historical | 0.686123 | 0.809278 | 0.652112 | 138 |
| scifact | 250 | Router_no_idf | 0.683188 | 0.808722 | 0.648200 | 97 |
| scifact | 250 | Router_no_margin | 0.657073 | 0.776833 | 0.623497 | 245 |
| nfcorpus | 50 | RRF_k10 | 0.334701 | 0.163868 | 0.541304 | 0 |
| nfcorpus | 50 | RRF_k100 | 0.329372 | 0.159326 | 0.540839 | 0 |
| nfcorpus | 50 | RRF_k30 | 0.329267 | 0.160215 | 0.538624 | 0 |
| nfcorpus | 50 | RRF_k60 | 0.329260 | 0.159291 | 0.540503 | 0 |
| nfcorpus | 50 | Router_always_hybrid | 0.329260 | 0.159291 | 0.540503 | 0 |
| nfcorpus | 50 | Router_deduplicated_calibration | 0.319704 | 0.154116 | 0.534261 | 95 |
| nfcorpus | 50 | Router_fixed_defaults | 0.326038 | 0.157569 | 0.541953 | 39 |
| nfcorpus | 50 | Router_historical | 0.319704 | 0.154116 | 0.534261 | 95 |
| nfcorpus | 50 | Router_no_idf | 0.325367 | 0.157092 | 0.540921 | 42 |
| nfcorpus | 50 | Router_no_margin | 0.312770 | 0.152349 | 0.524857 | 159 |
| nfcorpus | 250 | RRF_k10 | 0.336845 | 0.166030 | 0.543338 | 0 |
| nfcorpus | 250 | RRF_k100 | 0.329858 | 0.157431 | 0.537447 | 0 |
| nfcorpus | 250 | RRF_k30 | 0.331209 | 0.159363 | 0.539539 | 0 |
| nfcorpus | 250 | RRF_k60 | 0.330088 | 0.158444 | 0.537923 | 0 |
| nfcorpus | 250 | Router_always_hybrid | 0.330088 | 0.158444 | 0.537923 | 0 |
| nfcorpus | 250 | Router_deduplicated_calibration | 0.321360 | 0.154549 | 0.532455 | 95 |
| nfcorpus | 250 | Router_fixed_defaults | 0.327470 | 0.157645 | 0.539889 | 39 |
| nfcorpus | 250 | Router_historical | 0.321360 | 0.154549 | 0.532455 | 95 |
| nfcorpus | 250 | Router_no_idf | 0.326799 | 0.157168 | 0.538857 | 42 |
| nfcorpus | 250 | Router_no_margin | 0.313884 | 0.152718 | 0.524285 | 159 |
| fiqa | 50 | RRF_k10 | 0.356440 | 0.443808 | 0.425220 | 0 |
| fiqa | 50 | RRF_k100 | 0.342522 | 0.426649 | 0.414236 | 0 |
| fiqa | 50 | RRF_k30 | 0.346723 | 0.433632 | 0.418067 | 0 |
| fiqa | 50 | RRF_k60 | 0.342915 | 0.429221 | 0.413618 | 0 |
| fiqa | 50 | Router_always_hybrid | 0.342915 | 0.429221 | 0.413618 | 0 |
| fiqa | 50 | Router_deduplicated_calibration | 0.333803 | 0.416721 | 0.406292 | 66 |
| fiqa | 50 | Router_fixed_defaults | 0.341710 | 0.425363 | 0.414014 | 21 |
| fiqa | 50 | Router_historical | 0.333803 | 0.416721 | 0.406292 | 66 |
| fiqa | 50 | Router_no_idf | 0.341332 | 0.426392 | 0.413140 | 25 |
| fiqa | 50 | Router_no_margin | 0.265440 | 0.325297 | 0.336001 | 387 |
| fiqa | 250 | RRF_k10 | 0.360168 | 0.449116 | 0.428153 | 0 |
| fiqa | 250 | RRF_k100 | 0.334667 | 0.410712 | 0.409480 | 0 |
| fiqa | 250 | RRF_k30 | 0.349369 | 0.438428 | 0.419403 | 0 |
| fiqa | 250 | RRF_k60 | 0.338359 | 0.419739 | 0.411223 | 0 |
| fiqa | 250 | Router_always_hybrid | 0.338359 | 0.419739 | 0.411223 | 0 |
| fiqa | 250 | Router_deduplicated_calibration | 0.329346 | 0.406468 | 0.404233 | 66 |
| fiqa | 250 | Router_fixed_defaults | 0.337182 | 0.415881 | 0.411667 | 21 |
| fiqa | 250 | Router_historical | 0.329346 | 0.406468 | 0.404233 | 66 |
| fiqa | 250 | Router_no_idf | 0.336422 | 0.415367 | 0.410621 | 25 |
| fiqa | 250 | Router_no_margin | 0.265567 | 0.326200 | 0.336611 | 387 |

## Generation, citations, faithfulness and abstention

**New end-to-end RAG generation: BLOCKED.** The local Ollama endpoint refuses connections and Docker's daemon is unavailable (captured in `services.json` and `docker-version.txt`). In addition, BEIR qrels are relevance judgments, not reference answers, claim-level entailment annotations or answerability labels. A validated answer/claim test set and a specified, validated judge or documented human annotations are missing. No neural or human judgments were invented to fill these cells.

Inspection of `evaluate_real_llm_rag.py` shows direct calls to the generator with script-supplied contexts. It does not execute retrieval. Therefore the previous report's “end-to-end” characterization is unsupported. The mock-generation suite is a software fixture, not neural evidence. The archived Qwen outputs can support lexical/format recomputation only:

| Evidence | Cases | Answer quality | Citation support | Faithfulness | Abstention | Unsupported-claim rate |
|---|---|---|---|---|---|---|
| New retrieval → neural generation | 0 | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| Archived qwen2.5:0.5b supplied-context outputs | 10 | Token F1 0.379111 on 5 script-reference cases | No bracket citations on 5 answerable cases; entailment unverified | UNVERIFIED | 2/2 no-context fixtures detected; not general answerability | UNVERIFIED |
| Mock generator / hashed embeddings | Software tests only | Not benchmark evidence | Not benchmark evidence | Not benchmark evidence | Fixture behavior only | Not benchmark evidence |

Citation-ID presence is not citation entailment. The archive's abstention examples exercise a no-context fast path, not necessarily model reasoning. Script-source-ID checks and per-case lexical recomputations are in [generation_audit.json](../results/recomputed/generation_audit.json). No accuracy, human review, security robustness, or claim-support percentage is inferred from unsupported assertions. No confusion matrices are generated: rankings are not classification predictions, and no validated classification evaluation has been supplied.

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
