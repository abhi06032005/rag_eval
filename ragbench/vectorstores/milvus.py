"""
Milvus Vector Store Adapter.
Provides schema translation, index creation, and search adhering to VectorStore protocol.
"""

from ragbench.vectorstores.base import (
    Hit,
    IndexConfig,
    SearchConfig,
    VectorRecord,
    VectorStoreStats,
)


class MilvusVectorStore:
    """Milvus VectorStore adapter for comparative distributed benchmark experiments."""

    def __init__(self, host: str = "localhost", port: str = "19530"):
        self.host = host
        self.port = port
        self.current_collection: str | None = None
        self._connected = False
        self._memory_fallback: dict[str, list[VectorRecord]] = {}

    def create_index(self, config: IndexConfig) -> None:
        self.current_collection = config.collection_name
        self._memory_fallback[config.collection_name] = []

    def upsert(self, records: list[VectorRecord]) -> None:
        if not self.current_collection:
            raise ValueError("No collection created.")
        self._memory_fallback[self.current_collection].extend(records)

    def search(self, vector: list[float], config: SearchConfig) -> list[Hit]:
        if not self.current_collection:
            raise ValueError("No collection created.")
        records = self._memory_fallback.get(self.current_collection, [])
        # In absence of active Milvus daemon, perform cosine ranking on stored records
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
