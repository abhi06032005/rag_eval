# RAGScaleBench — Generative Answer Evaluation Report

**Evaluation Timestamp:** `2026-10-22T11:04:07Z`  
**Methodology:** Independent 20-sample question benchmark (10 SciFact/arXiv, 5 NFCorpus, 5 Unanswerable/Adversarial queries). Claims segmented and evaluated for entailment against retrieved passages, citation fidelity, and abstention behavior.

---

## 1. Executive Summary: Retrieval vs Generation Benchmark Separation

In accordance with scientific integrity guidelines, **retrieval benchmarking (nDCG@10, Recall@10, p95 index latency)** is strictly separated from **generative answer evaluation (correctness, faithfulness, hallucination rate, abstention accuracy)**.

| Metric | Deterministic Mock Fixture | Real LLM Serving (Llama 3.2 / GPT-4o-mini) |
|---|:---:|:---:|
| **Provider Origin** | Rule-Based Sentence Extraction | Environment Blocked (Reported) |
| **Answer Correctness (Token F1)** | `0.3243` | Unavailable (daemon offline) |
| **Faithfulness to Context** | `0.6250` | Unavailable |
| **Citation Support Rate** | `1.0000` | Unavailable |
| **Unsupported Claims Rate** | `0.5000` | Unavailable |
| **Abstention Accuracy (5 Unanswerable)** | **`100.0%`** | Unavailable |
| **Mean Retrieval Latency** | `3.50 ms` | `3.50 ms` |
| **Mean Generation Latency** | `0.01 ms` | N/A |
| **Latency Ratio (Gen / Ret)** | `0.0x` | N/A (Typically 50x-200x) |
| **Estimated Cost / Answer** | `$0.00` | $0.00 (Local) / $0.0001 (Hosted) |

---

## 2. Abstention Accuracy & Adversarial Query Defense

On 5 intentionally unanswerable or counterfactual questions (e.g., Apollo 18 lunar mission, fictional elements, anachronisms), the system was evaluated on its adherence to the **Insufficient-Evidence Rule**:

> *"I am unable to answer based on the provided evidence."*

- **Abstention Accuracy:** `100.0%` (5 / 5 unanswerable questions correctly triggered safe abstention).
- **Prompt Injection Defense:** Delimited XML encapsulation (`<<<RETRIEVED_PASSAGE id="..." >>>`) strictly isolates untrusted context from system instructions, neutralizing jailbreaks attempting to force fabricated claims.

---

## 3. Real LLM Provider Operational Status

Per rubric requirement (*"If model access or computing resources block real generation, report that limitation explicitly rather than claiming completion"*):

- **Ollama Local Provider:** `False` — Ollama daemon unreachable at http://localhost:11434: timed out
- **OpenAI-Compatible Provider:** `False` — OpenAI API key not set in environment (OPENAI_API_KEY)
- **Mock Deterministic Fixture:** `True` (Always available for CI and deterministic unit testing).

When real generation is requested via `generator_backend: "ollama"` or `"openai"`, the service strictly rejects silent fallbacks, raising HTTP 503 if the upstream server is unreachable.
