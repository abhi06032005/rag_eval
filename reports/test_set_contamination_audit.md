# Test-Set Contamination Audit & Protocol Integrity Verification
**RAGScaleBench — Phase 6 Research Deliverable (Task 5)**  
*Audit Date:* October 22, 2026  
*Status:* Completed Forensic Inspection & Protocol Stratification  

---

## 1. Executive Summary & Audit Mandate
A core requirement of empirical rigor in machine learning and information retrieval is maintaining strict separation between exploratory tuning and confirmatory testing. If hyperparameters are tuned against test query labels, reported metrics become overfitted and misleading.

This audit inspects every hyperparameter search conducted across Phases 4 and 5 of RAGScaleBench, specifically evaluating:
1. The **Reciprocal Rank Fusion (RRF) smoothing parameter ($k_{\text{rrf}}$) sweep**.
2. The **Top-$k$ cutoff sensitivity sweep**.
3. The **Adaptive Router confidence thresholds ($\tau_{\text{margin}}, \tau_{\text{idf}}$) calibration**.
4. The **Vector quantization rescoring search**.

---

## 2. Hyperparameter Search Audit Findings

| Investigation / Parameter | Target File | Query Split Used | Labels Accessed | Finding / Contamination Status | Formal Classification | Corrective Action Enforced |
|---|---|---|---|---|---|---|
| **RRF Parameter ($k_{\text{rrf}} \in [10, 20, 60, 100, 200]$)** | `scripts/run_stage5_ablations.py` | `test.tsv` (300 queries) | `test_qrels` | **Test-Set Reused Post-Hoc**: Sweep tested $k=10$ on locked test queries, yielding nDCG=0.6945. | **Exploratory Post-Hoc Finding** | Locked confirmatory baseline restricted strictly to Cormack et al. (2009) pre-specified $k_{\text{rrf}}=60$. $k=10$ labeled as exploratory sensitivity only. |
| **Top-$k$ Cutoff ($k \in [1, 3, 5, 10, 20, 50]$)** | `scripts/run_stage5_ablations.py` | `test.tsv` (300 queries) | `test_qrels` | **Sensitivity Check on Test**: Measured Recall & nDCG degradation across truncated ranked lists. | **Exploratory Ablation** | Documented as sensitivity analysis; not used to cherry-pick operating points. |
| **Adaptive Thresholds ($\tau_{\text{margin}}, \tau_{\text{idf}}$)** | `scripts/run_stage2_and_stage4_scifact.py` | `train.tsv` (809 queries) | `train_qrels` | **Clean Zero-Leakage Calibration**: Calibrated on independent training abstracts without accessing test split. | **Confirmatory Pre-Specified Protocol** | Confirmed valid. Locked parameters ($\tau_{\text{margin}}=0.20, \tau_{\text{idf}}=5.0$) evaluated on test split once without retuning. |
| **Adaptive Feature Ablation Grid** | `scripts/run_stage5_ablations.py` | `test.tsv` (300 queries) | `test_qrels` | **Post-Hoc Component Removal**: Evaluated removal of IDF, margin, and length heuristic on test set. | **Exploratory Feature Ablation** | Clearly labeled as post-hoc feature attribution rather than parameter optimization. |
| **NFCorpus Adaptive Thresholds** | `scripts/embed_and_evaluate_nfcorpus.py` | `dev.tsv` (324 queries) | `dev_qrels` | **Clean Zero-Leakage Calibration**: Calibrated strictly on official BEIR dev qrels. | **Confirmatory Pre-Specified Protocol** | Validated cross-domain transfer without touching test split. |
| **FiQA Adaptive Thresholds** | `scripts/embed_and_evaluate_fiqa.py` | `dev.tsv` (648 queries) | `dev_qrels` | **Clean Zero-Leakage Calibration**: Calibrated on financial dev set. | **Confirmatory Pre-Specified Protocol** | Evaluated on test queries without retuning. |

---

## 3. Detailed Audit of the RRF Parameter Sweep

### 3.1 The Contamination Mechanism
In `scripts/run_stage5_ablations.py` (lines 102–122):
```python
for k_val in [10, 20, 60, 100, 200]:
    for qid in query_ids: # query_ids is the locked 300 test queries
        gold = test_qrels[qid]
        fused = rrf_fuse(bm25_hits[qid], exact_hits[qid], k_rrf=k_val, top_k=50)
        ...
```
The resulting sweep table recorded:
- $k_{\text{rrf}}=10$: nDCG@10 = **0.6945**, Recall@10 = **0.8372**
- $k_{\text{rrf}}=20$: nDCG@10 = **0.6884**, Recall@10 = **0.8304**
- $k_{\text{rrf}}=60$: nDCG@10 = **0.6851**, Recall@10 = **0.8254**

If a researcher observed that $k_{\text{rrf}}=10$ produces an nDCG@10 of 0.6945 on `test.tsv` and subsequently selected $k=10$ as the project's headline baseline, that would constitute **test-set contamination**.

### 3.2 Corrective Action Enforced
1. **Preserve External Ground Truth:** In all canonical comparisons and the research paper, the headline Fixed Hybrid baseline remains strictly locked at **$k_{\text{rrf}}=60$**. This value is taken from Cormack, Clarke, and Büttcher (SIGIR 2009) as an external, un-tuned literature constant.
2. **Explicit Labeling:** The $k=10$ metric (0.6945) is archived in `results/canonical_results_registry.json` under status `EXPLORATORY_ABLATION`. The paper draft and website dashboard now explicitly state:
   > *"The parameter sweep $k_{\text{rrf}} \in [10, 20, 60, 100, 200]$ was conducted post-hoc on the test split. While $k=10$ improves agreement on SciFact, it represents exploratory parameter sensitivity rather than confirmatory evidence."*

---

## 4. Fresh Held-Out Evaluation Protocol & Nested Cross-Validation

To ensure complete statistical validity for confirmatory claims, we establish a standardized **Nested Evaluation Protocol**:

```
┌──────────────────────────────────────────────────────────────┐
│                    RAGScaleBench Split Protocol              │
├───────────────────────────────┬──────────────────────────────┤
│ Development Split (Train/Dev) │ Confirmatory Locked Test Set │
├───────────────────────────────┼──────────────────────────────┤
│ • SciFact: 809 train queries  │ • SciFact: 300 test queries  │
│ • NFCorpus: 324 dev queries   │ • NFCorpus: 323 test queries │
│ • FiQA: 648 dev queries       │ • FiQA: 648 test queries     │
│ • Tunes: tau_margin, tau_idf  │ • Evaluated ONCE at locked   │
│ • Zero test label access      │   thresholds. No retuning.   │
└───────────────────────────────┴──────────────────────────────┘
```

Under this protocol:
- **No threshold was tuned on test queries.**
- The Adaptive Router's performance on SciFact (nDCG = 0.6861) was obtained by directly applying the dev-selected thresholds ($\tau_{\text{margin}}=0.20, \tau_{\text{idf}}=5.0$) to the untouched test queries.
- Reused test sets are transparently declared and never presented as un-accessed splits.
