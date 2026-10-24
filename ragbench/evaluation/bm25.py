"""
Self-contained, Pure-Python BM25Okapi Lexical Search Engine.
High-performance inverted-index implementation with standard Okapi BM25 scoring.
Designed for reproducible information retrieval benchmarking without C-extension dependencies.
"""

import math
import re
from collections import Counter, defaultdict
from typing import Any


def tokenize_text(text: str) -> list[str]:
    """Fast regex alphanumeric tokenizer with lowercasing."""
    return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", text) if len(w) > 1]


class BM25Okapi:
    """
    Standard BM25Okapi algorithm with inverted indexing.
    k1: controls document term frequency saturation (default: 1.5).
    b: controls degree of document length normalisation (default: 0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_len: dict[str, int] = {}
        self.doc_freqs: dict[str, int] = defaultdict(int)
        self.inverted_index: dict[str, dict[str, int]] = defaultdict(dict)
        self.idf: dict[str, float] = {}

    def fit(self, documents: list[tuple[str, str]]) -> None:
        """
        Index a list of (doc_id, text) tuples.
        """
        self.corpus_size = len(documents)
        total_tokens = 0

        for doc_id, text in documents:
            tokens = tokenize_text(text)
            d_len = len(tokens)
            self.doc_len[doc_id] = d_len
            total_tokens += d_len

            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                self.inverted_index[term][doc_id] = count
                self.doc_freqs[term] += 1

        self.avgdl = total_tokens / max(self.corpus_size, 1)

        # Precompute Robertson-Spärck Jones IDF with smoothing
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """
        Score and rank corpus documents for a query.
        Returns list of (doc_id, score) sorted descending.
        """
        q_tokens = tokenize_text(query)
        if not q_tokens:
            return []

        scores: dict[str, float] = defaultdict(float)

        for term in q_tokens:
            if term not in self.inverted_index:
                continue
            term_idf = self.idf[term]
            postings = self.inverted_index[term]

            for doc_id, freq in postings.items():
                d_len = self.doc_len[doc_id]
                denom = freq + self.k1 * (1.0 - self.b + self.b * (d_len / self.avgdl))
                scores[doc_id] += term_idf * (freq * (self.k1 + 1.0)) / max(denom, 1e-9)

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
