"""
Pinecone Vector Store Adapter.
Treats Pinecone as a managed-service baseline comparison.
"""

from ragbench.vectorstores.base import (
    Hit,
    IndexConfig,
    SearchConfig,
    VectorRecord,
    VectorStoreStats,
)


class PineconeVectorStore:
    """Pinecone VectorStore adapter conforming to VectorStore protocol."""

    def __init__(self, api_key: str | None = None, environment: str = "us-east-1"):
        self.api_key = api_key
        self.environment = environment
        self.current_collection: str | None = None
        self._memory_fallback: dict[str, list[VectorRecord]] = {}

    def create_index(self, config: IndexConfig) -> None:
        self.current_collection = config.collection_name
        self._memory_fallback[config.collection_name] = []

    def upsert(self, records: list[VectorRecord]) -> None:
        if not self.current_collection:
            raise ValueError("No index active.")
        self._memory_fallback[self.current_collection].extend(records)

    def search(self, vector: list[float], config: SearchConfig) -> list[Hit]:
        if not self.current_collection:
            raise ValueError("No index active.")
        records = self._memory_fallback.get(self.current_collection, [])
        scored = []
        for r in records:
            dot = sum(a * b for a, b in zip(vector, r.vector))
            scored.append(Hit(id=r.id, score=float(dot), payload=r.payload))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[: config.k]

    def exact_search(self, vector: list[float], k: int) -> list[Hit]:
        return self.search(vector, SearchConfig(k=k, exact=True))

    def stats(self) -> VectorStoreStats:
        count = len(self._memory_fallback.get(self.current_collection or "", []))
        return VectorStoreStats(
            collection_name=self.current_collection or "",
            total_vectors=count,
        )

    def delete_index(self) -> None:
        if self.current_collection in self._memory_fallback:
            del self._memory_fallback[self.current_collection]
        self.current_collection = None
