"""
Production Qdrant Vector Store Implementation.
Manages isolated collections, HNSW indexing, quantisation modes, and exact search.
"""

import os
import time
import uuid

import psutil
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from ragbench.vectorstores.base import (
    Hit,
    IndexConfig,
    SearchConfig,
    VectorRecord,
    VectorStoreStats,
)


def to_qdrant_point_id(record_id: str) -> str:
    """Ensure record ID conforms to Qdrant UUID or integer requirement."""
    if record_id.isdigit():
        return record_id
    try:
        # Check if already a valid UUID
        return str(uuid.UUID(record_id))
    except (ValueError, AttributeError):
        # Deterministically convert any string ID into a UUIDv5
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, record_id))


class QdrantVectorStore:
    """
    Qdrant implementation exposing HNSW graph controls and vector quantisation.
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        prefer_grpc: bool = False,
        path: str | None = None,
        in_memory: bool = False,
    ):
        if in_memory:
            self.client = QdrantClient(location=":memory:")
        elif path:
            self.client = QdrantClient(path=path)
        elif url:
            self.client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)
        else:
            default_url = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333").replace("localhost", "127.0.0.1")
            try:
                self.client = QdrantClient(url=default_url, timeout=5.0)
                self.client.get_collections()
            except Exception:
                self.client = QdrantClient(location=":memory:")

        self.current_collection: str | None = None
        self.last_build_duration: float = 0.0

    def _map_distance(self, dist_str: str) -> rest.Distance:
        dist_str = dist_str.lower()
        if dist_str == "cosine":
            return rest.Distance.COSINE
        if dist_str == "dot":
            return rest.Distance.DOT
        if dist_str in ("euclidean", "euclid"):
            return rest.Distance.EUCLID
        return rest.Distance.COSINE

    def _map_quantization(self, mode: str, rescore: bool) -> rest.QuantizationConfig | None:
        mode = mode.lower()
        if mode == "scalar":
            return rest.ScalarQuantization(
                scalar=rest.ScalarQuantizationConfig(
                    type=rest.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            )
        if mode == "binary":
            return rest.BinaryQuantization(
                binary=rest.BinaryQuantizationConfig(
                    always_ram=True,
                )
            )
        if "product" in mode or "pq" in mode:
            return rest.ProductQuantization(
                product=rest.ProductQuantizationConfig(
                    compression=rest.CompressionRatio.X16,
                    always_ram=True,
                )
            )
        return None

    def create_index(self, config: IndexConfig) -> None:
        t0 = time.perf_counter()
        self.current_collection = config.collection_name
        distance = self._map_distance(config.distance)

        hnsw_config = rest.HnswConfigDiff(
            m=config.m,
            ef_construct=config.ef_construct,
            full_scan_threshold=1000,
        )

        quant_config = self._map_quantization(config.quantization, config.rescore)

        try:
            self.client.delete_collection(collection_name=config.collection_name)
        except Exception:
            pass

        self.client.create_collection(
            collection_name=config.collection_name,
            vectors_config=rest.VectorParams(
                size=config.dimension,
                distance=distance,
                on_disk=False,
            ),
            hnsw_config=hnsw_config,
            quantization_config=quant_config,
        )
        self.last_build_duration = time.perf_counter() - t0

    def upsert(self, records: list[VectorRecord]) -> None:
        if not self.current_collection:
            raise ValueError("No active collection. Call create_index first.")
        if not records:
            return

        t0 = time.perf_counter()
        points: list[rest.PointStruct] = []
        for r in records:
            pt_id = to_qdrant_point_id(r.id)
            payload = dict(r.payload)
            payload["_original_id"] = r.id
            payload["chunk_id"] = r.id
            points.append(
                rest.PointStruct(
                    id=pt_id,
                    vector=r.vector,
                    payload=payload,
                )
            )

        batch_size = 500
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.current_collection,
                points=batch,
                wait=True,
            )
        self.last_build_duration += time.perf_counter() - t0

    def search(self, vector: list[float], config: SearchConfig) -> list[Hit]:
        if not self.current_collection:
            env_coll = os.environ.get("QDRANT_COLLECTION")
            if env_coll:
                self.current_collection = env_coll
            else:
                try:
                    colls = self.client.get_collections().collections
                    if colls:
                        self.current_collection = colls[0].name
                except Exception:
                    pass
            if not self.current_collection:
                raise ValueError("No active collection. Call create_index first.")

        search_params = rest.SearchParams(
            hnsw_ef=config.hnsw_ef,
            exact=config.exact,
        )

        res = self.client.query_points(
            collection_name=self.current_collection,
            query=vector,
            limit=config.k,
            score_threshold=config.score_threshold,
            search_params=search_params,
            with_payload=True,
        )

        hits: list[Hit] = []
        for point in res.points:
            orig_id = (
                point.payload.get("_original_id", str(point.id)) if point.payload else str(point.id)
            )
            hits.append(
                Hit(
                    id=orig_id,
                    score=float(point.score),
                    payload=point.payload or {},
                )
            )
        return hits

    def exact_search(self, vector: list[float], k: int) -> list[Hit]:
        exact_cfg = SearchConfig(k=k, hnsw_ef=128, exact=True)
        return self.search(vector=vector, config=exact_cfg)

    def stats(self) -> VectorStoreStats:
        if not self.current_collection:
            env_coll = os.environ.get("QDRANT_COLLECTION")
            if env_coll:
                self.current_collection = env_coll
            else:
                try:
                    colls = self.client.get_collections().collections
                    if colls:
                        self.current_collection = colls[0].name
                except Exception:
                    pass
        if not self.current_collection:
            return VectorStoreStats(collection_name="", total_vectors=0)

        info = self.client.get_collection(collection_name=self.current_collection)
        total_pts = info.points_count or 0

        proc = psutil.Process(os.getpid())
        ram_bytes = proc.memory_info().rss

        return VectorStoreStats(
            collection_name=self.current_collection,
            total_vectors=total_pts,
            disk_size_bytes=total_pts * 384 * 4,
            ram_bytes=ram_bytes,
            index_build_seconds=self.last_build_duration,
        )

    def delete_index(self) -> None:
        if self.current_collection:
            try:
                self.client.delete_collection(collection_name=self.current_collection)
            except Exception:
                pass
            self.current_collection = None
