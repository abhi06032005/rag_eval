# RAGScaleBench

RAGScaleBench is a Python retrieval and RAG evaluation project with a FastAPI service and a Next.js dashboard. It includes BM25, exact dense search, reciprocal-rank fusion, an adaptive retrieval router, and Qdrant experiments.

## Evidence and limitations

The current independent audit is [FULL_BEIR_REEVALUATION.md](reports/FULL_BEIR_REEVALUATION.md). Its [environment and commands](reports/REPRODUCTION_ENVIRONMENT.md) and [machine-readable evidence](results/recomputed/) distinguish new runs, recomputed archives, mock fixtures, and blocked evaluations.

- Official SciFact, NFCorpus and FiQA archives are checked against BEIR's published MD5 values and locally pinned SHA-256 digests. SHA-256 values are not represented as upstream-published checksums.
- Retrieval uses official relevance judgments. NFCorpus has graded judgments; binary and graded nDCG must not be conflated.
- Candidate depth per branch, returned depth, and metric cutoff are separate settings. The former SciFact D50 registry entry incorrectly fused saved top-50 branches to represent a run that requested 250 candidates per branch. The corrected builder derives metrics from the appropriate rankings.
- Retrieval rankings do not define a classifier confusion matrix. No such matrix is claimed.
- Archived generation examples use script-authored references and supplied contexts. They do not establish end-to-end answer quality, citation entailment, faithfulness, or human-reviewed correctness.
- Unit tests use mock generation and offline embeddings where explicitly configured. In-memory Qdrant tests do not measure HNSW or quantization. The independent server benchmark refuses local emulation.
- Timings are observations on the recorded CPU host, with scope and contention limitations documented in the report. They are not service capacity or GPU measurements.

## Reproduce the audit

Use Python 3.11 or later and install the project and test dependencies:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e '.[dev]'
.venv/Scripts/python.exe -m scripts.full_beir_reevaluation --verify
```

The audit uses `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Download that revision into the Hugging Face cache before the offline model-loading step. The full environment records the versions actually used; unconstrained installation can select different versions.

```powershell
.venv/Scripts/python.exe -m scripts.full_beir_reevaluation --dataset scifact
.venv/Scripts/python.exe -m scripts.full_beir_reevaluation --dataset nfcorpus
.venv/Scripts/python.exe -m scripts.full_beir_reevaluation --dataset fiqa
```

These commands rebuild embeddings and lexical indexes, retain full candidate rankings, and compute per-query metrics and paired bootstrap statistics. Large arrays and serialized indexes are not committed; their checksums and regeneration commands are recorded.

For real ANN measurements, run the Qdrant version and isolated server configuration recorded in the environment report, then run `python -m scripts.run_recomputed_qdrant --dataset <dataset>`. Existing experiment collections are not overwritten. The report documents index readiness and actual indexed-vector counts.

The historical discrepancy reconstruction requires the archived embedding caches listed in its source manifest. It is a separate evidence class from the fresh runs:

```powershell
.venv/Scripts/python.exe -m scripts.recompute_scifact_discrepancy
.venv/Scripts/python.exe -m scripts.reconcile_and_build_canonical_registry
```

## Tests and web application

```powershell
$env:USE_OFFLINE_EMBEDDER='1'
.venv/Scripts/python.exe -m pytest tests/unit tests/integration
cd web
npm ci
npm run build
```

Tests require local socket support and tokenizer cache access. The audit also records the outcome of `npm run build -- --webpack` when the default compiler cannot start its worker. These commands are not evidence of neural answer quality.

## Project layout

| Path | Purpose |
|---|---|
| `ragbench/` | Retrieval, indexing, generation and evaluation components |
| `app/` | FastAPI service |
| `web/` | Dashboard source |
| `scripts/` | Benchmark and audit entry points |
| `tests/` | Unit and integration fixtures |
| `results/recomputed/` | Independent audit outputs and execution logs |
| `reports/` | Methods, findings and historical reports |

Historical reports remain available for traceability and may contain superseded claims. Use the independent audit and its explicit evidence classifications when interpreting results. Dataset attribution and licensing sources are listed in [DATA_LICENSES.md](DATA_LICENSES.md) and the new audit.
