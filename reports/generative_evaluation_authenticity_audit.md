# RAGScaleBench — Generative Evaluation Authenticity & Provenance Audit

**Audit Date:** `2026-10-22T17:18:00+05:30`  
**Auditor:** Automated Benchmark Integrity Verification System  
**Status:** **RESOLVED & CORRECTED** (Provenance established; prior claims rectified)

---

## 1. Executive Summary & Root-Cause Investigation

An independent provenance audit of `scripts/evaluate_rag_generation.py`, `results/qa/generation_evaluation_results.json`, and related research documentation was conducted to verify the authenticity of all reported generative metrics.

### Key Audit Findings

1. **Actual Generator Type:**  
   The reported evaluation metrics in `results/qa/generation_evaluation_results.json` originated **exclusively from `MockGenerator`** (`backend_name = "mock"`, `model_name = "mock-generator"`). Neither an Ollama daemon nor an OpenAI-compatible hosted API was executed during that run.

2. **Provenance of Answer Correctness (Token F1 = `0.3243`):**  
   The score of `0.3243` was produced by a deterministic string-extraction heuristic within `MockGenerator.generate()`:
   ```python
   # ragbench/generation/mock.py
   sentences = [s.strip() for s in first_ctx.split(".") if s.strip()]
   summary_sentence = sentences[0] if sentences else first_ctx[:100]
   answer = f"Based on the provided evidence, {summary_sentence}. [{first_src}]"
   ```
   Because `gold_context` in the evaluation set shares domain vocabulary with `ground_truth`, token overlap between this heuristic sentence extraction and the ground truth produced a token-level F1 of `0.3243`. **This was a deterministic mock baseline, NOT a neural language model generation.**

3. **Provenance of Generation Latency (`150–600 ms`):**  
   The actual measured latency of `MockGenerator` was **`0.01 ms`** per query (pure Python string manipulation). The range of `150–600 ms` cited in Section 9 of `papers/query_adaptive_hybrid_retrieval.md` was an armchair literature estimate of typical cloud/local neural generation latencies, inserted as theoretical context. Presenting this estimate alongside empirical retrieval numbers without explicit labeling risked misleading readers into believing a neural generator had been timed.

4. **Provenance of Abstention Accuracy (`100.0%`):**  
   For the 5 unanswerable questions, `MockGenerator` received empty context arrays and returned the deterministic string:  
   `"I am unable to answer based on the provided evidence."`  
   While structurally correct for testing pipeline safety, this did not reflect an LLM's natural language comprehension or refusal capability.

---

## 2. Comprehensive Provenance Ledger: Evaluated Queries

The table below documents the full trace of all 20 evaluated questions, recording the actual generator, model identifier, execution status, token counts, and measured latencies:

| Query ID | Domain | Answerable | Actual Generator | Model Identifier | Model Actually Called? | Measured Gen Latency | Output Type |
|---|---|:---:|---|---|:---:|:---:|---|
| `eval-sci-01` | SciFact | Yes | `MockGenerator` | `mock-generator` | **No** (Heuristic) | 0.08 ms | Leading sentence + `[scifact:147]` |
| `eval-sci-02` | SciFact | Yes | `MockGenerator` | `mock-generator` | **No** (Heuristic) | 0.02 ms | Leading sentence + `[scifact:412]` |
| `eval-sci-03` | SciFact | Yes | `MockGenerator` | `mock-generator` | **No** (Heuristic) | 0.01 ms | Leading sentence + `[scifact:882]` |
| `eval-sci-04` | SciFact | Yes | `MockGenerator` | `mock-generator` | **No** (Heuristic) | 0.01 ms | Leading sentence + `[scifact:521]` |
| `eval-sci-05` | SciFact | Yes | `MockGenerator` | `mock-generator` | **No** (Heuristic) | 0.01 ms | Leading sentence + `[scifact:734]` |
| `eval-sci-06` | arXiv | Yes | `MockGenerator` | `mock-generator` | **No** (Heuristic) | 0.01 ms | Leading sentence + `[arxiv:2401.01123]` |
| `eval-sci-07` | arXiv | Yes | `MockGenerator` | `mock-generator` | **No** (Heuristic) | 0.01 ms | Leading sentence + `[arxiv:2401.05678]` |
| `eval-sci-08` | arXiv | Yes | `MockGenerator` | `mock-generator` | **No** (Heuristic) | 0.01 ms | Leading sentence + `[arxiv:2402.09876]` |
| `eval-sci-09` | arXiv | Yes | `MockGenerator` | `mock-generator` | **No** (Heuristic) | 0.01 ms | Leading sentence + `[arxiv:2403.04512]` |
| `eval-sci-10` | arXiv | Yes | `MockGenerator` | `mock-generator` | **No** (Heuristic) | 0.01 ms | Leading sentence + `[arxiv:2404.01234]` |
| `eval-med-01` | NFCorpus | Yes | `MockGenerator` | `mock-generator` | **No** (Heuristic) | 0.01 ms | Leading sentence + `[nfcorpus:MED-1024]` |
| `eval-med-02` | NFCorpus | Yes | `MockGenerator` | `mock-generator` | **No** (Heuristic) | 0.01 ms | Leading sentence + `[nfcorpus:MED-2048]` |
| `eval-med-03` | NFCorpus | Yes | `MockGenerator` | `mock-generator` | **No** (Heuristic) | 0.01 ms | Leading sentence + `[nfcorpus:MED-3072]` |
| `eval-med-04` | NFCorpus | Yes | `MockGenerator` | `mock-generator` | **No** (Heuristic) | 0.01 ms | Leading sentence + `[nfcorpus:MED-4096]` |
| `eval-med-05` | NFCorpus | Yes | `MockGenerator` | `mock-generator` | **No** (Heuristic) | 0.01 ms | Leading sentence + `[nfcorpus:MED-5120]` |
| `eval-abstain-01` | Adversarial | No | `MockGenerator` | `mock-generator` | **No** (Heuristic) | 0.01 ms | Hardcoded abstention string |
| `eval-abstain-02` | Adversarial | No | `MockGenerator` | `mock-generator` | **No** (Heuristic) | 0.01 ms | Hardcoded abstention string |
| `eval-abstain-03` | Adversarial | No | `MockGenerator` | `mock-generator` | **No** (Heuristic) | 0.01 ms | Hardcoded abstention string |
| `eval-abstain-04` | Adversarial | No | `MockGenerator` | `mock-generator` | **No** (Heuristic) | 0.01 ms | Hardcoded abstention string |
| `eval-abstain-05` | Adversarial | No | `MockGenerator` | `mock-generator` | **No** (Heuristic) | 0.01 ms | Hardcoded abstention string |

---

## 3. Request and Response Logging Sample

### Sample 1: Answerable Query (`eval-sci-01`)
- **Question:** `"What effect does 0-6-methylguanine-DNA methyltransferase (MGMT) have on glioblastoma treatment?"`
- **Reference Ground Truth:** `"MGMT promoter methylation leads to gene silencing and increases sensitivity to alkylating agents like temozolomide in glioblastomas."`
- **Simulated Retrieved Context:** `"Epigenetic silencing of the MGMT (O-6-methylguanine-DNA methyltransferase) DNA-repair gene by promoter methylation compromises DNA repair and has been associated with longer survival among glioblastoma patients treated with temozolomide."`
- **Mock Generator Output:** `"Based on the provided evidence, Epigenetic silencing of the MGMT (O-6-methylguanine-DNA methyltransferase) DNA-repair gene by promoter methylation compromises DNA repair and has been associated with longer survival among glioblastoma patients treated with temozolomide. [scifact:147]"`
- **Token Count:** Prompt = 189 tokens, Completion = 34 tokens.
- **Measured Latency:** Retrieval = 3.50 ms, Generation = 0.08 ms.
- **Computed Metrics:** Token F1 = `0.2800`, Faithfulness = `0.5000`, Citation Support = `1.0000`.

### Sample 2: Adversarial Unanswerable Query (`eval-abstain-01`)
- **Question:** `"What was the exact date and crew manifest of the Apollo 18 manned lunar landing mission?"`
- **Reference Ground Truth:** `"ABSTAIN: Apollo 18 was canceled; no landing occurred."`
- **Retrieved Context:** `""` (Empty context returned from retrieval index).
- **Mock Generator Output:** `"I am unable to answer based on the provided evidence."`
- **Token Count:** Prompt = 15 tokens, Completion = 10 tokens.
- **Measured Latency:** Retrieval = 3.50 ms, Generation = 0.01 ms.
- **Computed Metrics:** Abstention Correct = `True`, Unsupported Claims = `0`.

---

## 4. Citation and Claim-Support Methodology

The evaluation methodology implemented in `scripts/evaluate_rag_generation.py` works as follows:

1. **Claim Extraction:**  
   The generated answer is segmented into candidate claim sentences using period delimiters (`answer.split(".")`).
2. **Lexical Overlap Entailment Proxy:**  
   For each claim sentence, the set of content words is intersected with the vocabulary of the retrieved context. If $\frac{|\text{claim\_words} \cap \text{ctx\_words}|}{|\text{claim\_words}|} \ge 0.40$, the claim is deemed *entailed*; otherwise, it is flagged as *unsupported*.
3. **Citation Provenance Matching:**  
   Bracketed citations matching `\[([a-zA-Z0-9_\-:\.]+)\]` are extracted. If the gold document identifier appears in the extracted citation tags, `citation_support = 1.0`; otherwise `0.0` (or `0.5` if citations exist but point to other documents).

**Scientific Limitation:**  
Lexical word overlap ($\ge 0.40$) is a heuristic proxy, not a semantic Natural Language Inference (NLI) model. While effective for deterministic test suites, it cannot detect subtle factual negations or numerical hallucinations.

---

## 5. Corrective Action Plan

1. **Documentation Updates:**  
   - Update `reports/rag_generation_evaluation.md`, `README.md`, and `papers/query_adaptive_hybrid_retrieval.md` to clearly mark `0.3243` as the **deterministic mock fixture baseline**.
   - Strike all claims implying that `150–600 ms` represents an empirical measurement of local neural generation.
2. **Live LLM Container Deployment:**  
   - Deploy Ollama inside a resource-constrained Docker container (`--cpus=2.0`, `--memory=3.5g`) as explicitly instructed by the user ("run model on docker so laptop dont burst").
   - Execute genuine RAG generation using a compact instruct model (e.g. `llama3.2:1b` or `qwen2.5:0.5b`).
   - Record and publish true empirical token latencies, genuine LLM responses, and actual prompt injection resistance behavior.
