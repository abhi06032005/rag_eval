"""
Ollama Generation Backend Adapter.
Supports local models (e.g. Llama 3.2 3B, Mistral, Qwen) with explicit error handling.
NEVER silently falls back to mock strings.
"""

from collections.abc import Iterator
import hashlib
import json
import os
import time

import httpx

from ragbench.generation.base import (
    GenerationChunk,
    GenerationResponse,
    build_rag_prompt,
    is_abstention_answer,
)


class OllamaConnectionError(RuntimeError):
    """Raised when Ollama daemon is unreachable or returns an unrecoverable error."""
    pass


class OllamaGenerator:
    """Ollama local serving adapter with strict real-LLM execution."""

    backend_name: str = "ollama"

    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        base_url: str | None = None,
        timeout: float = 45.0,
        max_retries: int = 2,
    ):
        self.model_name = model_name
        self.base_url = (
            base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def check_health(self) -> tuple[bool, str]:
        """Verify whether Ollama daemon is running and has the requested model."""
        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                if resp.status_code != 200:
                    return False, f"Ollama daemon returned HTTP {resp.status_code}"
                models = [m.get("name", "") for m in resp.json().get("models", [])]
                # Check model prefix or exact match
                has_model = any(self.model_name in m for m in models)
                if not has_model and models:
                    return False, f"Ollama running but model '{self.model_name}' not pulled. Available: {models[:3]}"
                return True, f"Ollama daemon healthy with model '{self.model_name}'"
        except Exception as exc:
            return False, f"Ollama daemon unreachable at {self.base_url}: {exc}"

    def generate(
        self,
        question: str,
        contexts: list[str],
        source_ids: list[str] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> GenerationResponse:
        t0 = time.perf_counter()
        prompt = build_rag_prompt(question, contexts, source_ids)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

        # Check for empty contexts -> prompt abstention directly without wasting LLM compute
        valid_contexts = [c.strip() for c in contexts if c and c.strip()]
        if not valid_contexts:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            return GenerationResponse(
                answer="I am unable to answer based on the provided evidence.",
                model_name=self.model_name,
                backend=self.backend_name,
                duration_ms=duration_ms,
                prompt_hash=prompt_hash,
                token_count=10,
                prompt_tokens=len(prompt.split()),
                completion_tokens=10,
                abstained=True,
                is_mock=False,
                finish_reason="abstention",
            )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 1.0,
                "num_predict": max_tokens,
            },
        }

        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(f"{self.base_url}/api/generate", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    answer = data.get("response", "").strip()
                    eval_count = data.get("eval_count", len(answer.split()))
                    prompt_eval_count = data.get("prompt_eval_count", len(prompt.split()))
                    duration_ns = data.get("total_duration")
                    duration_ms = (
                        (duration_ns / 1_000_000.0)
                        if duration_ns
                        else (time.perf_counter() - t0) * 1000.0
                    )
                    abstained = is_abstention_answer(answer)
                    return GenerationResponse(
                        answer=answer,
                        model_name=self.model_name,
                        backend=self.backend_name,
                        duration_ms=duration_ms,
                        prompt_hash=prompt_hash,
                        token_count=eval_count,
                        prompt_tokens=prompt_eval_count,
                        completion_tokens=eval_count,
                        abstained=abstained,
                        is_mock=False,
                        finish_reason="stop",
                        raw_response={"eval_duration": data.get("eval_duration")},
                    )
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(0.5 * (2**attempt))

        # Real LLM generation mode: DO NOT silently return mock string
        raise OllamaConnectionError(
            f"Ollama real generation failed after {self.max_retries + 1} attempts at {self.base_url}: {last_exc}. "
            "Real generation mode is active; silent mock fallback is strictly disabled."
        )

    def generate_stream(
        self,
        question: str,
        contexts: list[str],
        source_ids: list[str] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> Iterator[GenerationChunk]:
        """Stream generated tokens in real time from Ollama."""
        valid_contexts = [c.strip() for c in contexts if c and c.strip()]
        if not valid_contexts:
            yield GenerationChunk(
                token="I am unable to answer based on the provided evidence.",
                is_finished=True,
                finish_reason="abstention",
                token_count=10,
            )
            return

        prompt = build_rag_prompt(question, contexts, source_ids)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": 1.0,
                "num_predict": max_tokens,
            },
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as stream:
                    stream.raise_for_status()
                    token_acc = 0
                    for line in stream.iter_lines():
                        if not line:
                            continue
                        chunk_data = json.loads(line)
                        token_text = chunk_data.get("response", "")
                        is_done = chunk_data.get("done", False)
                        token_acc += 1
                        yield GenerationChunk(
                            token=token_text,
                            is_finished=is_done,
                            finish_reason="stop" if is_done else None,
                            token_count=token_acc,
                        )
                        if is_done:
                            break
        except Exception as exc:
            raise OllamaConnectionError(
                f"Ollama streaming connection failed at {self.base_url}: {exc}"
            )
