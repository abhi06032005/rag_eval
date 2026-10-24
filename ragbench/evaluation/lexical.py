"""
Lexical Answer-Correctness Metrics.
Computes Exact Match, Token F1, ROUGE-L, and BLEU.
Note: Lexical metrics assess answer similarity to reference, NOT context faithfulness.
"""

import re
import string
from collections import Counter
from collections.abc import Sequence


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def exact_match_score(
    prediction: str, ground_truth: str, aliases: Sequence[str] | None = None
) -> float:
    """Exact Match (EM) score (1.0 if normalized prediction matches truth or aliases, else 0.0)."""
    norm_pred = normalize_answer(prediction)
    candidates = [normalize_answer(ground_truth)]
    if aliases:
        candidates.extend(normalize_answer(a) for a in aliases)
    return 1.0 if norm_pred in candidates else 0.0


def token_f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 overlap score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return float(f1)


def rouge_l_score(prediction: str, ground_truth: str) -> float:
    """Compute ROUGE-L F1 score using longest common subsequence."""
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(ground_truth, prediction)
        return float(scores["rougeL"].fmeasure)
    except Exception:
        # Fallback LCS
        p_tokens = normalize_answer(prediction).split()
        g_tokens = normalize_answer(ground_truth).split()
        if not p_tokens or not g_tokens:
            return 0.0
        m, n = len(p_tokens), len(g_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if p_tokens[i] == g_tokens[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
        lcs = dp[m][n]
        if lcs == 0:
            return 0.0
        prec = lcs / m
        rec = lcs / n
        return float(2 * prec * rec / (prec + rec))


def bleu_score(prediction: str, ground_truth: str) -> float:
    """Compute BLEU score using token n-gram precision."""
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        p_tokens = normalize_answer(prediction).split()
        g_tokens = normalize_answer(ground_truth).split()
        if not p_tokens or not g_tokens:
            return 0.0
        smoother = SmoothingFunction().method1
        return float(sentence_bleu([g_tokens], p_tokens, smoothing_function=smoother))
    except Exception:
        return token_f1_score(prediction, ground_truth)
