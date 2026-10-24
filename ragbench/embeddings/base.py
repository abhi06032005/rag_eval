"""
Vector Embedder Protocol and Telemetry Base.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class EmbeddingTelemetry(BaseModel):
    model_name: str
    dimension: int
    batch_size: int
    duration_ms: float
    items_per_second: float


@runtime_checkable
class VectorEmbedder(Protocol):
    """Protocol defining the interface for all benchmark embedding models."""

    model_name: str
    dimension: int
    revision: str
    normalize: bool

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query."""
        ...

    def embed_passages(self, passages: list[str]) -> list[list[float]]:
        """Embed a batch of document passages."""
        ...
