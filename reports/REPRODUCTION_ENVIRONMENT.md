# Reproduction environment and commands

Execution date: 2026-10-24, Asia/Kolkata. Baseline repository commit: `1f9dad3`. Feature branch: `codex/full-beir-reevaluation`. No merge to main is part of this work.

## Hardware and runtime

- Windows build 10.0.26300, x86-64; AMD Ryzen 5 5500U, 6 physical cores / 12 logical processors.
- Physical memory: 16,456,007,680 bytes. No GPU benchmarking; installed PyTorch is CPU-only and CUDA availability was false.
- Python 3.11.15 in the existing project `.venv`; package versions are captured from installed distribution metadata in [environment.json](../results/recomputed/environment.json). `pip freeze` was unavailable because this environment has no pip module, so no successful freeze command is claimed.
- [requirements-reproduction.txt](../results/recomputed/requirements-reproduction.txt) is a metadata-derived dependency inventory, not a verified clean-install lock. Install the repository itself separately with `pip install -e . --no-deps` after provisioning compatible dependencies.
- Node v24.19.0; npm 11.16.0. Existing web dependencies were used, rather than claiming a clean `npm ci`. The repository's `web/package-lock.json` pins the tree.
- Qdrant server 1.19.2, commit `016542aa5deb6c66380bb137badf73d54f742bde`; Python client version is in the environment inventory. Binary: local `bin/qdrant.exe`. Isolated storage: `storage/recomputed-audit`, HTTP port 6335, gRPC 6336. The executable/storage are not committed.
- Local Ollama at `127.0.0.1:11434` refused connections. Docker CLI was installed but its daemon pipe was absent. Evidence: [services.json](../results/recomputed/services.json), [docker-version.txt](../results/recomputed/docker-version.txt).

## Model, input and numerical configuration

`sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, loaded with `local_files_only=True`, `device=cpu`. Cached model-file SHA-256 values are in [model_files_sha256.json](../results/recomputed/model_files_sha256.json). Model weights are not redistributed.

384-dimensional float32 normalized embeddings, maximum sequence length 256, document batch 64, query batch 32. Corpus input is concatenated title and text; corpus source order is retained and test query IDs are sorted. NumPy/PyTorch seeds 42; Torch intra-op threads 2, inter-op threads 1. Search shells set `OPENBLAS_NUM_THREADS=2`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`. Dense ties use stable input order in fresh runs and the original reverse-argsort order in the legacy reconstruction. Tiny batch/order-dependent query-embedding differences are audited separately.

Bootstrap uses `numpy.random.default_rng(42)`, 10,000 paired resamples, 95% percentile intervals and null-centered two-sided p-values with finite-resample correction. Main comparison families have 36 tests per dataset; calibrated replay and ANN have separate declared families. Qdrant graph construction is multithreaded, with no fixed server random seed exposed. No bitwise HNSW reproducibility is asserted.

## Exact commands and outcomes

Commands below were run from `D:\ragscalebench` unless a directory is specified. Output files record both failures and successful retries. `--dataset` commands were run separately for each of the three dataset names, not on subsamples.

### Repository and source review

```powershell
git fetch origin
git switch -c codex/full-beir-reevaluation
git show origin/feat/evidence-backed-evaluation-report-20261024:reports/EVALUATION_MATRIX_REPORT.md
```

The first sandboxed Git attempt could not write `.git`; the authorized unrestricted retry succeeded. The reviewed matrix report was absent on main, and found at feature commit `3137b2212da27eab40f26f0edad31a2a9daf0def`. Its text is saved under `results/recomputed/ARCHIVED_EVALUATION_MATRIX_REPORT.md`.

### Official dataset acquisition and verification

```powershell
.venv/Scripts/python.exe -m scripts.full_beir_reevaluation --verify
```

Initial sandbox attempt failed DNS access. The unrestricted retry downloaded all three archives over verified TLS and confirmed both local byte equality and BEIR's published MD5 digests. Extracted corpus/query/qrels SHA-256 hashes are checked against archive members. The script now pins the observed archive SHA-256 values and can populate a clean data directory. [Verification output](../results/recomputed/verification-output.txt), [manifest](../results/recomputed/dataset_verification.json).

For a fresh machine, acquire the pinned encoder revision first, for example by calling `huggingface_hub.snapshot_download('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41')`. This cache-provisioning command is an instruction for future reproduction; the audit used an existing cache whose bytes were hashed.

### Rebuild embeddings and run retrieval

```powershell
$env:OPENBLAS_NUM_THREADS='2'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
.venv/Scripts/python.exe -m scripts.full_beir_reevaluation --dataset scifact
.venv/Scripts/python.exe -m scripts.full_beir_reevaluation --dataset nfcorpus
.venv/Scripts/python.exe -m scripts.full_beir_reevaluation --dataset fiqa
```

Outputs: each dataset's `per_query.jsonl.gz`, `metrics.json`, `statistics.json`, `build.json`, ID lists and local embedding/index files under `results/recomputed/`. Run outputs are `{dataset}-run.txt`. Large `.npy` and `.pkl` files are excluded from Git; their hashes/sizes/build times are recorded. These commands really encode corpus vectors, rather than reusing the old arrays.

### Discrepancy, calibration and ablation audits

```powershell
.venv/Scripts/python.exe -m scripts.recompute_scifact_discrepancy
.venv/Scripts/python.exe -m scripts.replay_recomputed_ablations --dataset scifact
.venv/Scripts/python.exe -m scripts.replay_recomputed_ablations --dataset nfcorpus
.venv/Scripts/python.exe -m scripts.replay_recomputed_ablations --dataset fiqa
.venv/Scripts/python.exe -m scripts.check_embedding_drift
.venv/Scripts/python.exe -m scripts.audit_recomputed_generation
.venv/Scripts/python.exe -m scripts.reconcile_and_build_canonical_registry
.venv/Scripts/python.exe -m scripts.build_full_reevaluation_report
```

The discrepancy command uses the **archived** SciFact embedding cache and its ID files, checks reconstructed top-50 output against the committed archive, and stores its own reconstructed rankings. Archived caches are not part of a clean checkout; the independent fresh retrieval runs and committed reconstructed rankings provide separate reproducible evidence. The registry builder requires completed fresh outputs and the saved discrepancy rankings; it does not invent missing values. Calibration/feature/fusion runs replay saved candidate lists and make no new latency claims. `check_embedding_drift` compares available archived and new SciFact/NFCorpus vectors, and isolates changes caused by tie ordering.

### Real Qdrant server and comparison

```powershell
$auditProcess = Start-Process -FilePath 'D:/ragscalebench/bin/qdrant.exe' -ArgumentList '--config-path','D:/ragscalebench/results/recomputed/qdrant-config.yaml' -WorkingDirectory 'D:/ragscalebench' -WindowStyle Hidden -RedirectStandardOutput 'D:/ragscalebench/results/recomputed/qdrant-server.txt' -RedirectStandardError 'D:/ragscalebench/results/recomputed/qdrant-stderr.txt' -PassThru
$auditProcess.Id | Set-Content results/recomputed/qdrant-pid.txt
.venv/Scripts/python.exe -m scripts.run_recomputed_qdrant --dataset scifact
.venv/Scripts/python.exe -m scripts.run_recomputed_qdrant --dataset nfcorpus
.venv/Scripts/python.exe -m scripts.run_recomputed_qdrant --dataset fiqa
.venv/Scripts/python.exe -m scripts.recompute_ann_statistics --dataset scifact
.venv/Scripts/python.exe -m scripts.recompute_ann_statistics --dataset nfcorpus
.venv/Scripts/python.exe -m scripts.recompute_ann_statistics --dataset fiqa
```

Local socket access required execution outside the restricted sandbox. The first SciFact attempt was rejected by Qdrant because `full_scan_threshold=1` is below the server minimum; it was corrected to 10 and retried. Configuration snapshots verify GREEN status and full indexed-vector counts. Existing audit collections are not overwritten: a repeated experiment needs a fresh isolated storage directory/collection namespace. Audit-created server shutdown is recorded separately after measurement.

### Unit/integration tests

Initial attempts:

```powershell
.venv/Scripts/python.exe -m pytest tests/unit tests/integration --junitxml=results/recomputed/pytest.xml
$env:USE_OFFLINE_EMBEDDER='1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.venv/Scripts/python.exe -u -m pytest tests/unit tests/integration -vv -o faulthandler_timeout=30 --junitxml=results/recomputed/pytest.xml
```

Restricted attempts were interrupted when local event-loop socket creation stalled. The diagnostic output also showed tokenizer/temp-access failures; it does not represent a completed suite. Successful unrestricted retry:

```powershell
$env:USE_OFFLINE_EMBEDDER='1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.venv/Scripts/python.exe -u -m pytest tests/unit tests/integration -vv -o faulthandler_timeout=60 --junitxml=results/recomputed/pytest-retry.xml
```

**68 passed, 3 warnings, 57.62 seconds.** [Output](../results/recomputed/pytest-retry-output.txt). In-memory vector search and mock generator tests are software verification, not physical ANN/neural benchmarks. Final post-registry validation, if performed, is recorded in a separate final test log/XML.

### Production web build

Run in `D:\ragscalebench\web`:

```powershell
npm.cmd run build
npm.cmd run build
npm.cmd run build -- --webpack
```

1. Restricted default build: **FAIL**, Google Fonts fetch blocked.
2. Unrestricted default retry: **FAIL**, Turbopack CSS worker process exited before connection.
3. Unrestricted webpack production build: **PASS**, including TypeScript and static page generation.

Exact outputs: `web-build-output.txt`, `web-build-retry-output.txt`, `web-webpack-output.txt`; exit-code files are retained alongside them. No web source change was needed for the alternate build.

## Measurement limitations and retained evidence

The existing laptop was not dedicated to benchmarking. Embedding generation, test/build work and ANN work overlapped in portions of this audit. Timings are recorded observations with that interference, not isolated speedup claims. Search-only timings exclude query embedding and generation; serial QPS is not an offered-load capacity measurement. Process RSS includes Python/model/index overhead or multiple Qdrant collections. Full collection sizes include storage overhead, not just quantized vector payloads.

No human reviews, validated claim annotations, GPU measurements, thermal measurements or new neural answer metrics were performed. Missing generation endpoint and validated answer/claim labels are explicit blockers. Historical synthetic QA and mock fixtures are not substitutes for official relevance labels.
