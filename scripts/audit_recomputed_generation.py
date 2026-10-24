"""Audit archived generation records; lexical checks are not neural or human judgments."""

import json
from pathlib import Path

from ragbench.generation.base import is_abstention_answer
from scripts.evaluate_real_llm_rag import GENUINE_BENCHMARK_SET, compute_token_f1, extract_citations
from scripts.full_beir_reevaluation import OUT, dump, load_data, sha


def main():
    source = Path("results/qa/real_llm_evaluation_results.json")
    archive = json.loads(source.read_text())
    fixtures = {r["id"]: r for r in GENUINE_BENCHMARK_SET}
    rows = []
    corpora = {n: {r["_id"] for r in load_data(n)[0]} for n in ("scifact", "nfcorpus", "fiqa")}
    for r in archive["per_query_results"]:
        fixture = fixtures[r["id"]]
        answer = r["generated_answer"]
        cited = extract_citations(answer)
        ids = []
        for source_id in fixture["source_ids"]:
            domain, _, doc = source_id.partition(":")
            ids.append(
                {
                    "source_id": source_id,
                    "exists_in_official_corpus": doc in corpora[domain] if domain in corpora else None,
                }
            )
        rows.append(
            {
                "id": r["id"],
                "category": fixture["category"],
                "token_f1_recomputed": compute_token_f1(answer, fixture["ground_truth"]),
                "token_f1_archived": r["token_f1"],
                "bracket_citations": cited,
                "abstained_string_detector": is_abstention_answer(answer),
                "fixture_source_checks": ids,
                "evidence": "ARCHIVED_OUTPUT_RECOMPUTATION; script-authored reference and supplied context, not end-to-end retrieval",
            }
        )
    answerable = [r for r in rows if r["category"] in ("answerable", "multi_doc", "conflicting")]
    unanswerable = [r for r in rows if r["category"] == "unanswerable"]
    dump(
        OUT / "generation_audit.json",
        {
            "new_neural_generation": "BLOCKED",
            "missing_dependencies": [
                "No responsive Ollama model endpoint at 127.0.0.1:11434; Docker daemon unavailable",
                "BEIR retrieval qrels are not answer-quality, claim-support, or abstention labels; validated answer/claim annotation set and judge are absent",
            ],
            "archive_sha256": sha(source),
            "model": archive["model_label"],
            "archived_cases": len(rows),
            "answerable_cases": len(answerable),
            "token_f1_against_script_authored_reference": sum(r["token_f1_recomputed"] for r in answerable)
            / len(answerable),
            "explicit_citation_presence_rate": sum(bool(r["bracket_citations"]) for r in answerable) / len(answerable),
            "unanswerable_fixture_abstention_rate": sum(r["abstained_string_detector"] for r in unanswerable)
            / len(unanswerable),
            "faithfulness": None,
            "citation_entailment_support": None,
            "unsupported_claim_rate": None,
            "human_review": False,
            "confusion_matrix": None,
            "confusion_matrix_reason": "No validated classification labels/predictions supplied for scientific reporting",
            "per_case": rows,
        },
    )


if __name__ == "__main__":
    main()
