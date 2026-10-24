"""
Claim-Level Entailment and Groundedness Verification Module.
Decomposes answers into claims and classifies each as entailed, contradicted, or unsupported.
"""

import hashlib
import re

from pydantic import BaseModel


class ClaimResult(BaseModel):
    claim_text: str
    status: str  # entailed, unsupported, contradicted
    evidence_match_score: float


class EntailmentEvaluation(BaseModel):
    judge_model: str
    prompt_hash: str
    temperature: float = 0.0
    total_claims: int
    entailed_claims: int
    unsupported_claims: int
    contradicted_claims: int
    entailed_claim_rate: float
    unsupported_claim_rate: float
    contradicted_claim_rate: float
    claims: list[ClaimResult]


def decompose_into_claims(answer: str) -> list[str]:
    """Decompose generated answer into atomic factual sentences."""
    # Strip citation brackets first
    clean_ans = re.sub(r"\[[^\]]+\]", "", answer).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_ans) if len(s.strip()) > 10]
    return sentences if sentences else ([clean_ans] if clean_ans else [])


def evaluate_claim_entailment(
    answer: str,
    contexts: list[str],
    judge_model: str = "heuristic-nli-v1",
) -> EntailmentEvaluation:
    """
    Evaluate directional entailment of each claim in answer against retrieved contexts.
    """
    claims_text = decompose_into_claims(answer)
    prompt_hash = hashlib.sha256((" ".join(contexts) + answer).encode("utf-8")).hexdigest()[:16]

    if not claims_text or not contexts:
        return EntailmentEvaluation(
            judge_model=judge_model,
            prompt_hash=prompt_hash,
            total_claims=len(claims_text),
            entailed_claims=0,
            unsupported_claims=len(claims_text),
            contradicted_claims=0,
            entailed_claim_rate=0.0,
            unsupported_claim_rate=1.0 if claims_text else 0.0,
            contradicted_claim_rate=0.0,
            claims=[],
        )

    context_blob = " ".join(contexts).lower()
    claim_results: list[ClaimResult] = []

    entailed = 0
    unsupported = 0
    contradicted = 0

    for claim in claims_text:
        c_words = [w.lower().strip(".,;:\"'!?") for w in claim.split() if len(w) > 3]
        if not c_words:
            claim_results.append(
                ClaimResult(claim_text=claim, status="unsupported", evidence_match_score=0.0)
            )
            unsupported += 1
            continue

        matched = sum(1 for w in c_words if w in context_blob)
        score = float(matched) / float(len(c_words))

        # Check for direct negation contradictions
        has_negation = any(
            neg in claim.lower().split() for neg in ["never", "not", "no", "cannot", "failed"]
        )
        if score > 0.70:
            status = "entailed"
            entailed += 1
        elif score < 0.35:
            status = "unsupported"
            unsupported += 1
        else:
            status = "contradicted" if has_negation else "unsupported"
            if status == "contradicted":
                contradicted += 1
            else:
                unsupported += 1

        claim_results.append(
            ClaimResult(claim_text=claim, status=status, evidence_match_score=score)
        )

    total = len(claims_text)
    return EntailmentEvaluation(
        judge_model=judge_model,
        prompt_hash=prompt_hash,
        total_claims=total,
        entailed_claims=entailed,
        unsupported_claims=unsupported,
        contradicted_claims=contradicted,
        entailed_claim_rate=float(entailed) / float(total),
        unsupported_claim_rate=float(unsupported) / float(total),
        contradicted_claim_rate=float(contradicted) / float(total),
        claims=claim_results,
    )
