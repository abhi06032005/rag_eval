"""
Deterministic Mock Generator for CI, unit testing, and offline verification.
Clearly labeled as a mock heuristic fixture. Never claims to be an LLM.
"""

from collections.abc import Iterator
import hashlib
import time

from ragbench.generation.base import (
    GenerationChunk,
    GenerationResponse,
    build_rag_prompt,
    is_abstention_answer,
)


class MockGenerator:
    """Deterministic generator extracting leading sentences for offline test determinism."""

    backend_name: str = "mock"

    def __init__(self, model_name: str = "mock-generator"):
        self.model_name = model_name

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

        first_src = source_ids[0] if source_ids else "source:0"

        # Check for abstention: empty contexts
        valid_contexts = [c.strip() for c in contexts if c and c.strip()]
        if not valid_contexts:
            answer = "I am unable to answer based on the provided evidence."
            abstained = True
        else:
            first_ctx = valid_contexts[0]
            # Extract first sentence as key fact
            sentences = [s.strip() for s in first_ctx.split(".") if s.strip()]
            summary_sentence = sentences[0] if sentences else first_ctx[:100]
            answer = f"Based on the provided evidence, {summary_sentence}. [{first_src}]"
            abstained = is_abstention_answer(answer)

        duration_ms = (time.perf_counter() - t0) * 1000.0
        tokens = len(answer.split())

        return GenerationResponse(
            answer=answer,
            model_name=self.model_name,
            backend=self.backend_name,
            duration_ms=duration_ms,
            prompt_hash=prompt_hash,
            token_count=tokens,
            prompt_tokens=len(prompt.split()),
            completion_tokens=tokens,
            abstained=abstained,
            is_mock=True,
            finish_reason="stop",
        )

    def generate_stream(
        self,
        question: str,
        contexts: list[str],
        source_ids: list[str] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> Iterator[GenerationChunk]:
        """Stream mock response word-by-word for testing SSE endpoints."""
        valid_contexts = [c.strip() for c in contexts if c and c.strip()]
        if not valid_contexts:
            yield GenerationChunk(
                token="I am unable to answer based on the provided evidence.",
                is_finished=True,
                finish_reason="abstention",
                token_count=10,
            )
            return

        resp = self.generate(
            question=question,
            contexts=contexts,
            source_ids=source_ids,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        words = resp.answer.split(" ")
        token_acc = 0
        for i, w in enumerate(words):
            token_str = w if i == 0 else " " + w
            token_acc += 1
            is_last = (i == len(words) - 1)
            yield GenerationChunk(
                token=token_str,
                is_finished=is_last,
                finish_reason="stop" if is_last else None,
                token_count=token_acc,
            )


    def check_health(self) -> tuple[bool, str]:
        return True, "Mock generator fixture available (deterministic offline mode)"
