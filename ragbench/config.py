"""
RAGScaleBench Configuration Models and Loader.
Typed configuration definitions matching the benchmark specification.
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from ragbench.ids import compute_dict_sha256

VectorDBType = Literal["qdrant", "milvus", "pinecone"]
DistanceMetric = Literal["cosine", "dot", "euclidean"]
QuantizationMode = Literal["none", "scalar", "binary", "product_x16"]
GeneratorBackendType = Literal["ollama", "vllm", "api", "mock", "openai"]


class RepresentationStageConfig(BaseModel):
    embedding_models: list[str] = [
        "BAAI/bge-small-en-v1.5",
        "intfloat/e5-small-v2",
        "BAAI/bge-m3",
    ]
    chunk_size_tokens: list[int] = [128, 256, 448]
    overlap_tokens: list[int] = [0, 32]


class IndexStageConfig(BaseModel):
    hnsw_m: list[int] = [8, 16, 32]
    ef_construct: list[int] = [64, 128]
    quantization: list[QuantizationMode] = ["none", "scalar", "binary", "product_x16"]
    hnsw_ef: list[int] = [32, 64, 128]
    k: list[int] = [5, 10, 20]
    score_thresholds: list[str] = ["none", "dev_q05_gold", "dev_max_f1"]


class GenerationStageConfig(BaseModel):
    retrieval_selection: str = "pareto_top_6"
    generators: list[str] = ["llama3.2:3b", "llama3.1:8b"]
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 256


class LoadStageConfig(BaseModel):
    end_to_end_selection: str = "pareto_top_3"
    retrieve_rps: list[float] = [10.0, 25.0, 50.0, 100.0, 200.0]
    ask_rps: list[float] = [0.5, 1.0, 2.0, 5.0, 10.0]
    max_failure_rate: float = 0.01
    max_dropped_iteration_rate: float = 0.005
    max_latency_inflation_multiplier: float = 2.0


class BenchmarkConfig(BaseModel):
    """Single resolved configuration for a run or default pipeline."""

    total_documents: int = 75000
    arxiv_count: int = 30000
    wikipedia_count: int = 30000
    govinfo_count: int = 15000

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    chunk_size_tokens: int = 256
    overlap_tokens: int = 32

    vector_db: VectorDBType = "qdrant"
    distance: DistanceMetric = "cosine"
    hnsw_m: int = 16
    ef_construct: int = 128
    hnsw_ef: int = 64
    quantization: QuantizationMode = "none"
    pq_compression: int = 16
    k: int = 10
    score_threshold: str = "none"
    exact: bool = False

    generator_backend: GeneratorBackendType = "ollama"
    generator_model: str = "llama3.2:3b"
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 256

    dev_qa_path: str = "data/qa/dev.jsonl"
    test_qa_path: str = "data/qa/test.jsonl"

    @model_validator(mode="after")
    def validate_chunk_and_overlap(self) -> "BenchmarkConfig":
        if self.overlap_tokens < 0 or self.overlap_tokens >= self.chunk_size_tokens:
            raise ValueError(
                f"Invalid overlap ({self.overlap_tokens}): must be 0 <= overlap < chunk_size ({self.chunk_size_tokens})"
            )
        return self

    def config_hash(self) -> str:
        """Deterministic hash of this configuration."""
        return compute_dict_sha256(self.model_dump())[:16]


class GridExperimentsConfig(BaseModel):
    """Specification for staged experimental sweeps."""

    version: int = 1
    selection_metric: dict[str, Any] = Field(
        default_factory=lambda: {
            "primary": "retrieval_recall_at_10",
            "secondary": ["p95_retrieval_ms", "peak_qdrant_rss_mb"],
        }
    )
    representation: RepresentationStageConfig = Field(default_factory=RepresentationStageConfig)
    index: IndexStageConfig = Field(default_factory=IndexStageConfig)
    generation: GenerationStageConfig = Field(default_factory=GenerationStageConfig)
    load: LoadStageConfig = Field(default_factory=LoadStageConfig)


def load_config(path: str | Path) -> BenchmarkConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return BenchmarkConfig(**raw)


def load_grid_config(path: str | Path) -> GridExperimentsConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return GridExperimentsConfig(**raw)
