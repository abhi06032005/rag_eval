# Upstream Data Licences, Provenance & Attribution

This document records the exact provenance, upstream licensing terms, redistribution restrictions, and intellectual property attribution for all datasets, corpora, and source text incorporated into the **RAGScaleBench** benchmark.

---

## 1. Corpus License Matrix

| Corpus Domain | Snapshot Identifier | Upstream Authority / Source | Primary Licence | Redistribution Status | Permitted Commercial & Academic Use | Attribution Notice |
|---|---|---|---|---|---|---|
| **Scientific Pre-prints** | `arxiv-2026-10` | arXiv.org (Cornell University) | [arXiv Non-exclusive License to Distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) | Allowed (Metadata & Abstract) | Academic / Research Benchmarking | *"Metadata and abstracts provided courtesy of the arXiv API."* |
| **Encyclopedic Articles** | `enwiki-20261001` | Wikimedia Foundation | [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/) | Allowed with Attribution & ShareAlike | Yes, with attribution and downstream copyleft | *"Text from Wikipedia is available under the Creative Commons Attribution-ShareAlike License 4.0."* |
| **Federal Statutory & Regulatory** | `govinfo-bulk-2026` | U.S. Government Publishing Office | [Public Domain (17 U.S.C. § 105)](https://www.govinfo.gov/about/terms) | Allowed (No copyright protection within USA) | Unrestricted Worldwide | *"United States Government Work, free of copyright under 17 U.S.C. § 105."* |
| **UN Treaty Series (Optional)** | `un-treaty-v1` | United Nations Publications | UN Proprietary Permission Required | **Restricted / Excluded by Default** | Requires explicit written permission | Excluded until affirmative license reference is supplied |

---

## 2. Ingestion & Provenance Verification

- **Live arXiv Records:** Acquired via official arXiv REST API query endpoints (`http://export.arxiv.org/api/query`) adhering to the compulsory 3.0-second delay between HTTP requests. Only abstracts and metadata are stored; full binary PDFs are not republished.
- **Live Wikipedia Articles:** Ingested via MediaWiki API action queries (`https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1`). Raw source records preserve original Page IDs, revisions, and URLs.
- **GovInfo Fallback Fixtures:** Due to high-volume bulk XML quotas and API key restrictions, GovInfo text in the initial snapshot is programmatically generated legislative fixture text modeled directly upon real Code of Federal Regulations and Federal Register circular formats.
- **Content Integrity:** The combined corpus manifest is cryptographically pinned via SHA-256 digests in [`data/data_manifest.lock.yaml`](file:///d:/ragscalebench/data/data_manifest.lock.yaml).

---

## 3. Evaluation Questions Provenance

- **Development Set (100 QA pairs):** Programmatically generated via factual statement decomposition and evidence span matching.
- **Locked Test Set (50 QA pairs):** Generated via deterministic token offset alignment. In adherence to research ethics:
  > **Note on Human Verification:** In accordance with the project audit, the 50 test questions are classified as **synthetic/unverified**. No manual human verification is claimed. All evidence spans are character-aligned to underlying source documents.

---

## 4. Software Dependencies Attribution

This project incorporates the following open-source libraries:
- **FastAPI**: MIT License, © Sebastián Ramírez
- **Qdrant Client**: Apache License 2.0, © Qdrant Solutions GmbH
- **Sentence-Transformers**: Apache License 2.0, © UKP Lab / Hugging Face
- **PyTorch**: Modified BSD License, © Meta Platforms, Inc. & PyTorch Contributors
- **Pydantic**: MIT License, © Samuel Colvin
- **Tiktoken**: MIT License, © OpenAI
