"""
OpenAI-Compatible Generation Backend Adapter.
Supports OpenAI, vLLM, DeepSeek, LocalAI, and any OpenAI-compatible serving engine.
Includes real streaming (SSE), retry backoff, and strict no-silent-fallback guarantees.
"""

from collections.abc import Iterator
import hashlib
import json
import os
import time

import httpx

from ragbench.generation.base import (
    SYSTEM_PROMPT,
    GenerationChunk,
    GenerationResponse,
    build_rag_prompt,
    is_abstention_answer,
)


class OpenAIConnectionError(RuntimeError):
    """Raised when the OpenAI-compatible endpoint is unreachable or returns an error."""
    pass


class OpenAIGenerator:
    """Adapter for hosted or local OpenAI-compatible endpoints."""

    backend_name: str = "openai"

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 45.0,
        max_retries: int = 2,
    ):
        self.model_name = model_name
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("API_GENERATOR_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("API_GENERATOR_API_KEY", "")
        )
        self.timeout = timeout
        self.max_retries = max_retries

    def check_health(self) -> tuple[bool, str]:
        """Verify endpoint connectivity."""
        if not self.api_key and "localhost" not in self.base_url and "127.0.0.1" not in self.base_url:
            return False, "OpenAI API key not set in environment (OPENAI_API_KEY)"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            with httpx.Client(timeout=2.0) as client:
                r = client.get(f"{self.base_url}/models", headers=headers)
                if r.status_code == 200:
                    return True, f"OpenAI-compatible endpoint reachable at {self.base_url}"
                return False, f"Endpoint returned HTTP {r.status_code}"
        except Exception as exc:
            return False, f"Endpoint unreachable at {self.base_url}: {exc}"

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

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    choice = data["choices"][0]
                    answer = choice["message"]["content"].strip()
                    usage = data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", len(prompt.split()))
                    comp_tokens = usage.get("completion_tokens", len(answer.split()))
                    finish_reason = choice.get("finish_reason", "stop")
                    duration_ms = (time.perf_counter() - t0) * 1000.0
                    abstained = is_abstention_answer(answer)

                    return GenerationResponse(
                        answer=answer,
                        model_name=self.model_name,
                        backend=self.backend_name,
                        duration_ms=duration_ms,
                        prompt_hash=prompt_hash,
                        token_count=comp_tokens,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=comp_tokens,
                        abstained=abstained,
                        is_mock=False,
                        finish_reason=finish_reason,
                        raw_response=usage,
                    )
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(0.5 * (2**attempt))

        raise OpenAIConnectionError(
            f"OpenAI generation failed after {self.max_retries + 1} attempts at {self.base_url}: {last_exc}. "
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
        """Stream chat completion tokens in real-time using SSE."""
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
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST", f"{self.base_url}/chat/completions", json=payload, headers=headers
                ) as stream:
                    stream.raise_for_status()
                    token_acc = 0
                    for line in stream.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        chunk_obj = json.loads(data_str)
                        choices = chunk_obj.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        token_text = delta.get("content", "")
                        finish_reason = choices[0].get("finish_reason")
                        is_last = finish_reason is not None
                        if token_text or is_last:
                            token_acc += 1
                            yield GenerationChunk(
                                token=token_text,
                                is_finished=is_last,
                                finish_reason=finish_reason,
                                token_count=token_acc,
                            )
                        if is_last:
                            break
        except Exception as exc:
            raise OpenAIConnectionError(
                f"OpenAI streaming failed at {self.base_url}: {exc}"
            )
