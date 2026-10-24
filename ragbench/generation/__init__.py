"""
RAG Generation Backend Factory and Registry.
Supports Mock, Ollama, OpenAI-compatible (vLLM, DeepSeek, Hosted), and strict no-silent-fallback guarantees.
"""

from typing import Any

from ragbench.generation.api import ApiGenerator
from ragbench.generation.base import (
    SYSTEM_PROMPT,
    GenerationChunk,
    GenerationResponse,
    GeneratorBackend,
    GeneratorMetadata,
    build_rag_prompt,
    is_abstention_answer,
    sanitize_context_text,
)
from ragbench.generation.mock import MockGenerator
from ragbench.generation.ollama import OllamaConnectionError, OllamaGenerator
from ragbench.generation.openai import OpenAIConnectionError, OpenAIGenerator
from ragbench.generation.vllm import VllmGenerator

__all__ = [
    "SYSTEM_PROMPT",
    "GenerationChunk",
    "GenerationResponse",
    "GeneratorBackend",
    "GeneratorMetadata",
    "build_rag_prompt",
    "is_abstention_answer",
    "sanitize_context_text",
    "MockGenerator",
    "OllamaGenerator",
    "OllamaConnectionError",
    "OpenAIGenerator",
    "OpenAIConnectionError",
    "VllmGenerator",
    "ApiGenerator",
    "get_generator",
    "list_generator_providers",
]


def get_generator(
    backend: str = "mock",
    model_name: str | None = None,
    **kwargs: Any,
) -> GeneratorBackend:
    """
    Factory function to instantiate generation backends.
    
    Args:
        backend: One of 'mock', 'ollama', 'openai', 'vllm', 'api'.
        model_name: Specific model identifier (defaults appropriately per backend).
        **kwargs: Extra parameters passed to the backend constructor (base_url, api_key, etc.).
    """
    backend_norm = backend.lower().strip()
    if backend_norm == "ollama":
        model = model_name or "llama3.2:3b"
        return OllamaGenerator(model_name=model, **kwargs)
    elif backend_norm in ("openai", "vllm", "deepseek"):
        model = model_name or "gpt-4o-mini"
        return OpenAIGenerator(model_name=model, **kwargs)
    elif backend_norm == "api":
        model = model_name or "api-model"
        return ApiGenerator(model_name=model, **kwargs)
    elif backend_norm == "mock":
        model = model_name or "mock-generator"
        return MockGenerator(model_name=model)
    else:
        raise ValueError(
            f"Unsupported generation backend '{backend}'. "
            "Supported backends: 'mock', 'ollama', 'openai', 'vllm', 'api'."
        )


def list_generator_providers() -> list[dict[str, Any]]:
    """Return health and availability status of all configured generator providers."""
    providers = []

    # Mock
    mock_gen = MockGenerator()
    ok_mock, msg_mock = mock_gen.check_health()
    providers.append({
        "backend": "mock",
        "default_model": "mock-generator",
        "runtime_mode": "mock_fixture",
        "is_mock": True,
        "available": ok_mock,
        "message": msg_mock,
    })

    # Ollama
    try:
        ollama_gen = OllamaGenerator()
        ok_ollama, msg_ollama = ollama_gen.check_health()
    except Exception as exc:
        ok_ollama, msg_ollama = False, str(exc)
    providers.append({
        "backend": "ollama",
        "default_model": "llama3.2:3b",
        "runtime_mode": "real_llm",
        "is_mock": False,
        "available": ok_ollama,
        "message": msg_ollama,
        "base_url": getattr(ollama_gen, "base_url", "http://localhost:11434"),
    })

    # OpenAI / OpenAI-compatible
    try:
        openai_gen = OpenAIGenerator()
        ok_openai, msg_openai = openai_gen.check_health()
    except Exception as exc:
        ok_openai, msg_openai = False, str(exc)
    providers.append({
        "backend": "openai",
        "default_model": "gpt-4o-mini",
        "runtime_mode": "real_llm",
        "is_mock": False,
        "available": ok_openai,
        "message": msg_openai,
        "base_url": getattr(openai_gen, "base_url", "https://api.openai.com/v1"),
    })

    return providers
