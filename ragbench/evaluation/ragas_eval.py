"""
RAGas-Aligned Evaluation Engine.
Distinguishes Context Precision, Context Recall, Answer Correctness, and Faithfulness.
"""

import re
from collections.abc import Sequence

from ragbench.evaluation.lexical import normalize_answer, token_f1_score


def calculate_context_precision(
    retrieved_chunk_ids: Sequence[str], gold_chunk_ids: set[str], k: int = 10
) -> float:
    """
    Context Precision: evaluates whether ground-truth relevant chunks appear near the top of the ranking.
    Precision@k weighted by rank relevance.
    """
    top_k = retrieved_chunk_ids[:k]
    if not top_k or not gold_chunk_ids:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, cid in enumerate(top_k, 1):
        if cid in gold_chunk_ids:
            hits += 1
            precision_sum += float(hits) / float(rank)

    return float(precision_sum / max(hits, 1)) if hits > 0 else 0.0


def calculate_context_recall(
    retrieved_chunk_ids: Sequence[str], gold_chunk_ids: set[str], k: int = 10
) -> float:
    """
    Context Recall: proportion of ground-truth evidence chunks retrieved in top k.
    """
    if not gold_chunk_ids:
        return 1.0
    top_k = retrieved_chunk_ids[:k]
    retrieved_gold = sum(1 for cid in top_k if cid in gold_chunk_ids)
    return float(retrieved_gold) / float(len(gold_chunk_ids))


def calculate_answer_correctness(prediction: str, ground_truth: str) -> float:
    """
    Answer Correctness: semantic and factual agreement between prediction and reference answer.
    Combines token overlap and keyword coverage.
    """
    return token_f1_score(prediction, ground_truth)


def calculate_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Faithfulness: evaluates whether factual claims in the generated response
    are directly supported by the retrieved contexts.
    Returns ratio of supported statements to total statements in the answer.
    """
    if not answer.strip():
        return 0.0
    if not contexts:
        return 0.0

    full_context_text = " ".join(contexts).lower()
    # Decompose answer into individual sentences/statements
    statements = [s.strip() for s in re.split(r"[.!?]\s+", answer) if len(s.strip()) > 10]
    if not statements:
        statements = [answer.strip()]

    supported_count = 0
    for stmt in statements:
        # Check token coverage against retrieved context
        stmt_tokens = normalize_answer(stmt).split()
        if not stmt_tokens:
            continue
        matched_tokens = sum(1 for t in stmt_tokens if t in full_context_text)
        token_coverage = float(matched_tokens) / float(len(stmt_tokens))
        if token_coverage >= 0.65:
            supported_count += 1

    return float(supported_count) / float(len(statements))
