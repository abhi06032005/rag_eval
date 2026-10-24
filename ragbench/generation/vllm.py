"""
vLLM OpenAI-Compatible Generation Backend Adapter.
"""

import hashlib
import os
import time

import httpx

from ragbench.generation.base import (
    SYSTEM_PROMPT,
    GenerationResponse,
    build_rag_prompt,
)


class VllmGenerator:
    """vLLM GPU Serving Adapter."""

    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model_name = model_name
        self.base_url = (
            base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")

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

        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Contexts:\n{' '.join(contexts)}\n\nQuestion: {question}",
                },
            ],
            "temperature": temperature,
            "top_p": 1.0,
            "max_tokens": max_tokens,
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                answer = choice["message"]["content"].strip()
                eval_count = data.get("usage", {}).get("completion_tokens", len(answer.split()))
        except Exception as exc:
            first_src = source_ids[0] if source_ids else "source:0"
            answer = f"[vLLM Unavailable: {exc}] Context excerpt: {contexts[0][:120] if contexts else ''} [{first_src}]"
            eval_count = len(answer.split())

        duration_ms = (time.perf_counter() - t0) * 1000.0
        return GenerationResponse(
            answer=answer,
            model_name=self.model_name,
            duration_ms=duration_ms,
            prompt_hash=prompt_hash,
            token_count=eval_count,
        )
