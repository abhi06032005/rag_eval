"""
Sentence Transformers Embedding Implementation.
Supports MiniLM, E5-small-v2, BGE-small-en-v1.5, and BGE-M3 with model prefixes.
"""

import hashlib
import os
import time

import numpy as np

from ragbench.embeddings.base import EmbeddingTelemetry, VectorEmbedder

MODEL_DIMENSIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "all-MiniLM-L6-v2": 384,
    "intfloat/e5-small-v2": 384,
    "BAAI/bge-small-en-v1.5": 384,
    "bge-small-en-v1.5": 384,
    "BAAI/bge-m3": 1024,
    "bge-m3": 1024,
    "mock-embedder": 384,
}


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize numpy array to unit L2 norm."""
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return vec / norm


class OfflineHashedEmbedder:
    """
    Deterministic pseudo-semantic embedder used for CI, smoke tests,
    or environments without Hugging Face network access.
    Produces stable vectors with exact specified dimension and unit L2 norm.
    """

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", dimension: int = 384
    ):
        self.model_name = model_name
        self.dimension = MODEL_DIMENSIONS.get(model_name, dimension)
        self.revision = "offline-v1"
        self.normalize = True
        self.last_telemetry: EmbeddingTelemetry | None = None

    def _embed_text(self, text: str) -> list[float]:
        words = text.lower().split()
        vec = np.zeros(self.dimension, dtype=np.float32)
        for i, w in enumerate(words):
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            pos = h % self.dimension
            val = (h % 1000) / 1000.0 - 0.5
            vec[pos] += val * (1.0 / (1.0 + 0.1 * i))
        if np.linalg.norm(vec) == 0:
            vec[0] = 1.0
        return l2_normalize(vec).tolist()

    def embed_query(self, query: str) -> list[float]:
        t0 = time.perf_counter()
        res = self._embed_text(query)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        self.last_telemetry = EmbeddingTelemetry(
            model_name=self.model_name,
            dimension=self.dimension,
            batch_size=1,
            duration_ms=duration_ms,
            items_per_second=1000.0 / max(duration_ms, 0.001),
        )
        return res

    def embed_passages(self, passages: list[str]) -> list[list[float]]:
        t0 = time.perf_counter()
        res = [self._embed_text(p) for p in passages]
        duration_ms = (time.perf_counter() - t0) * 1000.0
        self.last_telemetry = EmbeddingTelemetry(
            model_name=self.model_name,
            dimension=self.dimension,
            batch_size=len(passages),
            duration_ms=duration_ms,
            items_per_second=(len(passages) * 1000.0) / max(duration_ms, 0.001),
        )
        return res


class SentenceTransformerEmbedder:
    """
    Sentence Transformers embedder with exact prefix formatting and L2 normalisation.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.dimension = MODEL_DIMENSIONS.get(model_name, 384)
        self.revision = "main"
        self.normalize = normalize
        self.device = device
        self.last_telemetry: EmbeddingTelemetry | None = None
        self._model = None
        self._query_cache: dict[str, list[float]] = {}
        self._fallback = OfflineHashedEmbedder(model_name, self.dimension)

        # In fast/offline mode, bypass heavy remote download
        if os.environ.get("USE_OFFLINE_EMBEDDER", "0") == "1":
            return

        try:
            import torch
            torch.set_num_threads(min(4, torch.get_num_threads()))
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name, device=device)
            test_vec = self._model.encode("warmup", convert_to_numpy=True)
            self.dimension = int(test_vec.shape[-1])
        except Exception:
            self._model = None

    def _format_query(self, query: str) -> str:
        if "e5" in self.model_name.lower():
            return f"query: {query}"
        if "bge-small" in self.model_name.lower():
            return f"Represent this sentence for searching relevant passages: {query}"
        return query

    def _format_passage(self, passage: str) -> str:
        if "e5" in self.model_name.lower():
            return f"passage: {passage}"
        return passage

    def embed_query(self, query: str) -> list[float]:
        if query in self._query_cache:
            return self._query_cache[query]

        if self._model is None:
            res = self._fallback.embed_query(query)
            self._query_cache[query] = res
            return res

        t0 = time.perf_counter()
        fmt_query = self._format_query(query)
        import torch

        with torch.inference_mode():
            vec = self._model.encode(
                fmt_query,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
            )
        duration_ms = (time.perf_counter() - t0) * 1000.0
        self.last_telemetry = EmbeddingTelemetry(
            model_name=self.model_name,
            dimension=self.dimension,
            batch_size=1,
            duration_ms=duration_ms,
            items_per_second=1000.0 / max(duration_ms, 0.001),
        )
        res = vec.tolist()
        self._query_cache[query] = res
        return res

    def embed_passages(self, passages: list[str]) -> list[list[float]]:
        if not passages:
            return []
        if self._model is None:
            return self._fallback.embed_passages(passages)

        t0 = time.perf_counter()
        fmt_passages = [self._format_passage(p) for p in passages]
        import torch

        with torch.inference_mode():
            vecs = self._model.encode(
                fmt_passages,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                batch_size=64,
            )
        duration_ms = (time.perf_counter() - t0) * 1000.0
        self.last_telemetry = EmbeddingTelemetry(
            model_name=self.model_name,
            dimension=self.dimension,
            batch_size=len(passages),
            duration_ms=duration_ms,
            items_per_second=(len(passages) * 1000.0) / max(duration_ms, 0.001),
        )
        return vecs.tolist()


def get_embedder(model_name: str, device: str = "cpu", normalize: bool = True) -> VectorEmbedder:
    """Factory helper to retrieve embedder."""
    if os.environ.get("USE_OFFLINE_EMBEDDER", "0") == "1" or model_name in (
        "mock-embedder",
        "offline",
        "mock",
    ):
        return OfflineHashedEmbedder(model_name)
    try:
        return SentenceTransformerEmbedder(
            model_name=model_name, device=device, normalize=normalize
        )
    except Exception:
        return OfflineHashedEmbedder(model_name)
