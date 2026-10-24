"""
Vector Store Protocol and Unified Data Models.
"""

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class VectorRecord(BaseModel):
    id: str
    vector: list[float]
    payload: dict[str, Any] = Field(default_factory=dict)


class Hit(BaseModel):
    id: str
    score: float
    payload: dict[str, Any] = Field(default_factory=dict)


class IndexConfig(BaseModel):
    collection_name: str
    dimension: int
    distance: str = "cosine"  # cosine, dot, euclidean
    m: int = 16
    ef_construct: int = 128
    quantization: str = "none"  # none, scalar, binary, product_x16
    pq_compression: int = 16
    rescore: bool = True


class SearchConfig(BaseModel):
    k: int = 10
    hnsw_ef: int = 64
    score_threshold: float | None = None
    exact: bool = False


class VectorStoreStats(BaseModel):
    collection_name: str
    total_vectors: int
    disk_size_bytes: int = 0
    ram_bytes: int = 0
    index_build_seconds: float = 0.0


@runtime_checkable
class VectorStore(Protocol):
    """Unified Vector Store Protocol across Qdrant, Milvus, and Pinecone."""

    def create_index(self, config: IndexConfig) -> None: ...

    def upsert(self, records: list[VectorRecord]) -> None: ...

    def search(self, vector: list[float], config: SearchConfig) -> list[Hit]: ...

    def exact_search(self, vector: list[float], k: int) -> list[Hit]:
        """Bypass HNSW graph to perform exact brute-force vector search."""
        ...

    def stats(self) -> VectorStoreStats: ...

    def delete_index(self) -> None: ...
