# QA Benchmark Evaluation Validity & Leakage Report

**Report Date:** October 2026  
**Evaluation Splits:** Development Set (`data/qa/dev.jsonl`, 100 items), Locked Test Set (`data/qa/test.jsonl`, 50 items)  
**External Benchmark:** SciFact Adapter (`data/external/scifact_corpus.parquet`, 5 items)  
**Repository:** RAGScaleBench (EDISS Portfolio)  

---

## 1. Executive Summary

This report assesses the experimental validity of the question–answering evaluation benchmarks in **RAGScaleBench**. In strict alignment with research integrity standards:
1. We distinguish **programmatic span validation** (verifying that character offsets cleanly map to source documents) from **factual verification** (verifying that questions and answers reflect genuine real-world semantics).
2. We identify and measure an **information leakage vector** (document title presence in query strings).
3. We publish an independent human review sheet ([`reports/test_set_human_review_sheet.md`](file:///d:/ragscalebench/reports/test_set_human_review_sheet.md)) and classify the test set as `unverified_programmatic` until human sign-off occurs.
4. We integrate an established external open-access benchmark (**SciFact**, Allen Institute for AI) to benchmark retrieval without title leakage.

---

## 2. Evidence Offset & Document Association Verification

All 150 question–answer records were evaluated against the processed document corpus (`data/processed/documents.parquet`):

| Evaluation Set | Record Count | Source Domain Distribution | Verbatim Offset Match Rate | Unmapped Questions across Chunks (128 / 256 / 448 tokens) |
|---|---|---|---|---|
| **Development (`dev.jsonl`)** | 100 | arXiv (79), GovInfo (21) | **100 / 100 (100.0%)** | **0** (100% mapped across all configurations) |
| **Locked Test (`test.jsonl`)** | 50 | arXiv (20), Wikipedia (20), GovInfo (10) | **50 / 50 (100.0%)** | **0** (100% mapped across all configurations) |

- **Offset Integrity:** Every `evidence_spans[0]` (`start_char`, `end_char`) directly matches the substring in the corresponding canonical document under UTF-8 decoding with zero character drift.
- **Dynamic Interval Mapping:** The interval intersection algorithm (`derive_relevant_chunk_ids`) maps evidence spans to 1.92 chunks on average for 128-token chunking, 1.34 chunks for 256-token chunking, and 1.02 chunks for 448-token chunking.

---

## 3. Programmatic Span Validation vs. Factual Verification

A crucial distinction in IR and RAG benchmarking:
- **Programmatic Span Validation (Passed):** Confirms that ground-truth answers exist in the corpus and that evidence span coordinates point to identical text in the database.
- **Factual Verification (Pending Human Sign-off):**
  - The 100 dev and 50 test questions were generated via automated sentence extraction templates (`"According to study '{title}', what finding or methodology is established?"`).
  - While grammatically valid, these queries are synthetic templates.
  - In accordance with our audit findings, the 50 test items have been updated to:
    ```json
    "creation_method": "synthetic_template",
    "review_status": "unverified_programmatic"
    ```
  - **No manual human verification is claimed.**

---

## 4. Information Leakage Vector Analysis

- **The Title Leakage Vector:**
  - **Dev Set:** 99.0% of questions explicitly contain the exact document title in the question text.
  - **Test Set:** 100.0% of questions explicitly contain the exact document title in the question text.
- **Experimental Impact:**
  Because vector chunk text includes the document title, dense retrieval models (such as BGE and MiniLM) can exploit exact title token overlap rather than performing deeper semantic matching.
- **Mitigation & External Benchmark:**
  To evaluate semantic retrieval capability independently of title leakage, we introduced the **SciFact** benchmark adapter ([`ragbench/data/external_benchmarks.py`](file:///d:/ragscalebench/ragbench/data/external_benchmarks.py)). SciFact queries are natural scientific claims (e.g. *"Anti-PD-1 immunotherapy produces objective responses in patients with advanced solid tumors"*) that contain zero title leakage tokens.

---

## 5. Independent Human Review Sheet

A review sheet has been generated at [`reports/test_set_human_review_sheet.md`](file:///d:/ragscalebench/reports/test_set_human_review_sheet.md). It displays all 50 locked test items with:
- Target document locator and title.
- Question and ground-truth answer.
- Exact evidence span excerpt.
- Interactive rubric checkboxes for factual accuracy, title independence, and verdict sign-off.
