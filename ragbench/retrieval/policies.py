"""
Modular Retrieval Policies and Query-Adaptive Quality-Aware Router.
Supports:
  A. BM25 Only
  B. Dense Only
  C. Fixed Hybrid RRF (BM25 + Dense)
  D. Adaptive Quality-Aware Router
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass, field
import time
from typing import Any, Callable
import numpy as np

from ragbench.evaluation.bm25 import BM25Okapi, tokenize_text


def rrf_fuse(
    ranked_a: list[tuple[str, float]],
    ranked_b: list[tuple[str, float]],
    k_rrf: int = 60,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion of two ranked candidate lists."""
    rrf_scores: dict[str, float] = defaultdict(float)
    for rank, (doc_id, _) in enumerate(ranked_a, 1):
        rrf_scores[doc_id] += 1.0 / (k_rrf + rank)
    for rank, (doc_id, _) in enumerate(ranked_b, 1):
        rrf_scores[doc_id] += 1.0 / (k_rrf + rank)
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


@dataclass
class RetrievalResult:
    retrieved_docs: list[tuple[str, float]]
    route_taken: str  # "bm25" | "dense" | "hybrid"
    reason: str
    features: dict[str, float] = field(default_factory=dict)
    dense_search_executed: bool = False
    bm25_search_executed: bool = False
    latency_ms: float = 0.0
    neural_embedding_invocations: int = 0
    dense_search_requests: int = 0
    timings_breakdown_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RetrievalPolicy(ABC):
    @abstractmethod
    def retrieve(
        self,
        query_text: str,
        query_vec: np.ndarray | None = None,
        top_k: int = 10,
    ) -> RetrievalResult:
        """Execute retrieval policy for a query."""
        pass


class BM25Policy(RetrievalPolicy):
    def __init__(self, bm25_engine: BM25Okapi):
        self.bm25 = bm25_engine

    def retrieve(
        self,
        query_text: str,
        query_vec: np.ndarray | None = None,
        top_k: int = 10,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        hits = self.bm25.search(query_text, top_k=top_k)
        dur_ms = (time.perf_counter() - t0) * 1000.0
        return RetrievalResult(
            retrieved_docs=hits,
            route_taken="bm25",
            reason="fixed_policy_bm25_only",
            bm25_search_executed=True,
            dense_search_executed=False,
            latency_ms=dur_ms,
            neural_embedding_invocations=0,
            dense_search_requests=0,
            timings_breakdown_ms={
                "feature_extraction_ms": 0.0,
                "bm25_search_ms": round(dur_ms, 3),
                "query_embedding_ms": 0.0,
                "dense_search_ms": 0.0,
                "rrf_fusion_ms": 0.0,
                "total_ms": round(dur_ms, 3),
            },
        )


class DensePolicy(RetrievalPolicy):
    def __init__(
        self,
        dense_search_fn: Callable[[np.ndarray, int], list[tuple[str, float]]],
        embed_query_fn: Callable[[str], np.ndarray] | None = None,
    ):
        """
        dense_search_fn accepts (query_vec, top_k) and returns list of (doc_id, score).
        embed_query_fn accepts query_text and returns query_vec.
        """
        self.dense_search_fn = dense_search_fn
        self.embed_query_fn = embed_query_fn

    def retrieve(
        self,
        query_text: str,
        query_vec: np.ndarray | None = None,
        top_k: int = 10,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        emb_ms = 0.0
        neural_calls = 0

        if query_vec is None:
            if self.embed_query_fn is not None:
                t_emb = time.perf_counter()
                query_vec = self.embed_query_fn(query_text)
                emb_ms = (time.perf_counter() - t_emb) * 1000.0
                neural_calls = 1
            else:
                raise ValueError("query_vec or embed_query_fn is required for DensePolicy")
        else:
            neural_calls = 1  # pre-computed by caller

        t_dense = time.perf_counter()
        hits = self.dense_search_fn(query_vec, top_k)
        dense_ms = (time.perf_counter() - t_dense) * 1000.0
        dur_ms = (time.perf_counter() - t0) * 1000.0

        return RetrievalResult(
            retrieved_docs=hits,
            route_taken="dense",
            reason="fixed_policy_dense_only",
            bm25_search_executed=False,
            dense_search_executed=True,
            latency_ms=dur_ms,
            neural_embedding_invocations=neural_calls,
            dense_search_requests=1,
            timings_breakdown_ms={
                "feature_extraction_ms": 0.0,
                "bm25_search_ms": 0.0,
                "query_embedding_ms": round(emb_ms, 3),
                "dense_search_ms": round(dense_ms, 3),
                "rrf_fusion_ms": 0.0,
                "total_ms": round(dur_ms, 3),
            },
        )


class FixedHybridPolicy(RetrievalPolicy):
    def __init__(
        self,
        bm25_engine: BM25Okapi,
        dense_search_fn: Callable[[np.ndarray, int], list[tuple[str, float]]],
        embed_query_fn: Callable[[str], np.ndarray] | None = None,
        k_rrf: int = 60,
        candidate_depth: int | None = None,
    ):
        self.bm25 = bm25_engine
        self.dense_search_fn = dense_search_fn
        self.embed_query_fn = embed_query_fn
        self.k_rrf = k_rrf
        if candidate_depth is not None and candidate_depth < 10:
            raise ValueError("candidate_depth must be at least 10")
        self.candidate_depth = candidate_depth

    def retrieve(
        self,
        query_text: str,
        query_vec: np.ndarray | None = None,
        top_k: int = 10,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        t_bm = time.perf_counter()
        depth = self.candidate_depth if self.candidate_depth is not None else max(top_k * 5, 50)
        if depth < top_k:
            raise ValueError("candidate_depth must be >= top_k")
        bm25_hits = self.bm25.search(query_text, top_k=depth)
        bm25_ms = (time.perf_counter() - t_bm) * 1000.0

        emb_ms = 0.0
        neural_calls = 0
        if query_vec is None:
            if self.embed_query_fn is not None:
                t_emb = time.perf_counter()
                query_vec = self.embed_query_fn(query_text)
                emb_ms = (time.perf_counter() - t_emb) * 1000.0
                neural_calls = 1
            else:
                raise ValueError("query_vec or embed_query_fn is required for FixedHybridPolicy")
        else:
            neural_calls = 1

        t_dense = time.perf_counter()
        dense_hits = self.dense_search_fn(query_vec, depth)
        dense_ms = (time.perf_counter() - t_dense) * 1000.0

        t_rrf = time.perf_counter()
        fused = rrf_fuse(bm25_hits, dense_hits, k_rrf=self.k_rrf, top_k=top_k)
        rrf_ms = (time.perf_counter() - t_rrf) * 1000.0
        dur_ms = (time.perf_counter() - t0) * 1000.0

        return RetrievalResult(
            retrieved_docs=fused,
            route_taken="hybrid",
            reason="fixed_policy_hybrid_rrf",
            bm25_search_executed=True,
            dense_search_executed=True,
            latency_ms=dur_ms,
            neural_embedding_invocations=neural_calls,
            dense_search_requests=1,
            timings_breakdown_ms={
                "feature_extraction_ms": 0.0,
                "bm25_search_ms": round(bm25_ms, 3),
                "query_embedding_ms": round(emb_ms, 3),
                "dense_search_ms": round(dense_ms, 3),
                "rrf_fusion_ms": round(rrf_ms, 3),
                "total_ms": round(dur_ms, 3),
            },
        )


@dataclass
class AdaptiveThresholds:
    tau_lexical_margin: float = 0.28  # BM25 top margin (s1 - s2)/s1
    tau_min_idf: float = 6.0         # Rare technical term presence
    tau_dense_confident: float = 0.75 # High dense cosine similarity
    tau_bm25_weak: float = 8.0       # Weak lexical score threshold
    k_rrf: int = 60


class AdaptiveRouterPolicy(RetrievalPolicy):
    """
    Query-Adaptive, Quality-Aware Retrieval Policy.
    Uses pre-retrieval lexical signals (query length, max IDF) and 
    initial BM25 score margins to bypass expensive dense neural retrieval 
    when lexical confidence is high.
    Falls back to Dense or Hybrid RRF under uncertainty or low lexical confidence.
    """

    def __init__(
        self,
        bm25_engine: BM25Okapi,
        dense_search_fn: Callable[[np.ndarray, int], list[tuple[str, float]]],
        thresholds: AdaptiveThresholds | None = None,
        embed_query_fn: Callable[[str], np.ndarray] | None = None,
        candidate_depth: int | None = None,
    ):
        self.bm25 = bm25_engine
        self.dense_search_fn = dense_search_fn
        self.embed_query_fn = embed_query_fn
        self.cfg = thresholds or AdaptiveThresholds()
        if candidate_depth is not None and candidate_depth < 10:
            raise ValueError("candidate_depth must be at least 10")
        self.candidate_depth = candidate_depth

    def extract_lexical_signals(self, query_text: str) -> dict[str, float]:
        tokens = tokenize_text(query_text)
        q_len = float(len(tokens))
        idfs = [self.bm25.idf.get(t, 0.0) for t in tokens]
        max_idf = float(max(idfs)) if idfs else 0.0
        mean_idf = float(np.mean(idfs)) if idfs else 0.0
        return {
            "query_len": q_len,
            "max_idf": round(max_idf, 4),
            "mean_idf": round(mean_idf, 4),
        }

    def retrieve(
        self,
        query_text: str,
        query_vec: np.ndarray | None = None,
        top_k: int = 10,
    ) -> RetrievalResult:
        t0 = time.perf_counter()

        # 0. Pre-retrieval feature extraction
        t_feat = time.perf_counter()
        signals = self.extract_lexical_signals(query_text)
        feat_ms = (time.perf_counter() - t_feat) * 1000.0

        # 1. First-stage lexical probe (fast inverted index lookup)
        t_bm = time.perf_counter()
        depth = self.candidate_depth if self.candidate_depth is not None else max(top_k * 5, 50)
        if depth < top_k:
            raise ValueError("candidate_depth must be >= top_k")
        bm25_hits = self.bm25.search(query_text, top_k=depth)
        bm25_ms = (time.perf_counter() - t_bm) * 1000.0

        bm25_top_score = bm25_hits[0][1] if bm25_hits else 0.0
        bm25_runner_up = bm25_hits[1][1] if len(bm25_hits) > 1 else 0.0
        bm25_margin = (
            (bm25_top_score - bm25_runner_up) / (bm25_top_score + 1e-6)
            if bm25_top_score > 0
            else 0.0
        )

        signals["bm25_top_score"] = round(bm25_top_score, 4)
        signals["bm25_margin"] = round(bm25_margin, 4)

        # Rule 1: High Lexical Confidence & Rare Entity Signal
        # If BM25 has a wide score separation margin and contains specific high-IDF terms,
        # lexical matching is authoritative. Both neural embedding and dense search are skipped.
        if (
            signals["bm25_margin"] >= self.cfg.tau_lexical_margin
            and signals["max_idf"] >= self.cfg.tau_min_idf
            and signals["bm25_top_score"] >= self.cfg.tau_bm25_weak
        ):
            dur_ms = (time.perf_counter() - t0) * 1000.0
            return RetrievalResult(
                retrieved_docs=bm25_hits[:top_k],
                route_taken="bm25",
                reason="high_lexical_margin_and_rare_terms",
                features=signals,
                bm25_search_executed=True,
                dense_search_executed=False,
                latency_ms=dur_ms,
                neural_embedding_invocations=0,
                dense_search_requests=0,
                timings_breakdown_ms={
                    "feature_extraction_ms": round(feat_ms, 3),
                    "bm25_search_ms": round(bm25_ms, 3),
                    "query_embedding_ms": 0.0,
                    "dense_search_ms": 0.0,
                    "rrf_fusion_ms": 0.0,
                    "total_ms": round(dur_ms, 3),
                },
            )

        # If lexical confidence is insufficient, invoke dense branch
        emb_ms = 0.0
        neural_calls = 0
        if query_vec is None:
            if self.embed_query_fn is not None:
                t_emb = time.perf_counter()
                query_vec = self.embed_query_fn(query_text)
                emb_ms = (time.perf_counter() - t_emb) * 1000.0
                neural_calls = 1
            else:
                raise ValueError("query_vec or embed_query_fn is required when dense retrieval is triggered")
        else:
            neural_calls = 1

        t_dense = time.perf_counter()
        dense_hits = self.dense_search_fn(query_vec, depth)
        dense_ms = (time.perf_counter() - t_dense) * 1000.0

        dense_top_score = dense_hits[0][1] if dense_hits else 0.0
        signals["dense_top_score"] = round(dense_top_score, 4)

        # Rule 2: Dense Semantic Dominance
        # When lexical matches are completely absent or weak, but dense similarity is strong
        if (
            dense_top_score >= self.cfg.tau_dense_confident
            and bm25_top_score < (self.cfg.tau_bm25_weak * 0.5)
        ):
            dur_ms = (time.perf_counter() - t0) * 1000.0
            return RetrievalResult(
                retrieved_docs=dense_hits[:top_k],
                route_taken="dense",
                reason="dense_semantic_dominance_weak_lexical",
                features=signals,
                bm25_search_executed=True,
                dense_search_executed=True,
                latency_ms=dur_ms,
                neural_embedding_invocations=neural_calls,
                dense_search_requests=1,
                timings_breakdown_ms={
                    "feature_extraction_ms": round(feat_ms, 3),
                    "bm25_search_ms": round(bm25_ms, 3),
                    "query_embedding_ms": round(emb_ms, 3),
                    "dense_search_ms": round(dense_ms, 3),
                    "rrf_fusion_ms": 0.0,
                    "total_ms": round(dur_ms, 3),
                },
            )

        # Rule 3: Low-Confidence Fallback / Rank Disagreement -> Hybrid RRF
        t_rrf = time.perf_counter()
        fused = rrf_fuse(bm25_hits, dense_hits, k_rrf=self.cfg.k_rrf, top_k=top_k)
        rrf_ms = (time.perf_counter() - t_rrf) * 1000.0
        dur_ms = (time.perf_counter() - t0) * 1000.0

        return RetrievalResult(
            retrieved_docs=fused,
            route_taken="hybrid",
            reason="uncertainty_fallback_hybrid_rrf",
            features=signals,
            bm25_search_executed=True,
            dense_search_executed=True,
            latency_ms=dur_ms,
            neural_embedding_invocations=neural_calls,
            dense_search_requests=1,
            timings_breakdown_ms={
                "feature_extraction_ms": round(feat_ms, 3),
                "bm25_search_ms": round(bm25_ms, 3),
                "query_embedding_ms": round(emb_ms, 3),
                "dense_search_ms": round(dense_ms, 3),
                "rrf_fusion_ms": round(rrf_ms, 3),
                "total_ms": round(dur_ms, 3),
            },
        )
