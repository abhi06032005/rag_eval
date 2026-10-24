"""
Dependency Injection Container for Service and Experiment Runner.
Provides unified instances of Embedder, VectorStore, and Generator.
"""

import os

from ragbench.config import BenchmarkConfig, load_config
from ragbench.embeddings.sentence_transformers import VectorEmbedder, get_embedder
from ragbench.generation import GeneratorBackend, get_generator
from ragbench.vectorstores.base import VectorStore
from ragbench.vectorstores.qdrant import QdrantVectorStore


class RAGPipelineComponents:
    """Singleton container holding production pipeline services."""

    def __init__(self, config_path: str = "configs/default.yaml"):
        if os.path.exists(config_path):
            self.config = load_config(config_path)
        else:
            self.config = BenchmarkConfig()

        # Initialize Embedder
        self.embedder: VectorEmbedder = get_embedder(
            model_name=self.config.embedding_model,
            device=os.environ.get("EMBEDDING_DEVICE", "cpu"),
        )

        # Initialize VectorStore
        self.vector_store: VectorStore = QdrantVectorStore()

        # Initialize Generator (defaults to mock for deterministic testing unless overridden)
        backend = os.environ.get("GENERATOR_BACKEND", "mock")
        model = os.environ.get("GENERATOR_MODEL", "mock-generator")
        self.generator: GeneratorBackend = get_generator(
            backend=backend,
            model_name=model,
        )



_components: RAGPipelineComponents | None = None


def get_pipeline_components() -> RAGPipelineComponents:
    global _components
    if _components is None:
        _components = RAGPipelineComponents()
    return _components


def get_generator_instance(
    backend: str | None = None,
    model_name: str | None = None,
) -> GeneratorBackend:
    """Return requested generator or fallback to singleton default."""
    components = get_pipeline_components()
    if not backend and not model_name:
        return components.generator
    
    selected_backend = backend or os.environ.get("GENERATOR_BACKEND", components.config.generator_backend)
    return get_generator(backend=selected_backend, model_name=model_name)

