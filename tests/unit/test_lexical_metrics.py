from ragbench.evaluation.entailment import evaluate_claim_entailment
from ragbench.evaluation.lexical import (
    exact_match_score,
    token_f1_score,
)
from ragbench.evaluation.ragas_eval import calculate_faithfulness


def test_exact_match_known_examples():
    truth = "Product Quantisation"
    assert exact_match_score("product quantisation", truth) == 1.0
    assert exact_match_score("The Product Quantisation.", truth) == 1.0
    assert exact_match_score("Scalar Quantisation", truth) == 0.0


def test_token_f1_known_examples():
    truth = "vector quantisation for dense retrieval"
    pred_exact = "vector quantisation for dense retrieval"
    pred_partial = "vector quantisation"

    assert token_f1_score(pred_exact, truth) == 1.0
    assert 0.0 < token_f1_score(pred_partial, truth) < 1.0
    assert token_f1_score("completely unrelated words", truth) == 0.0


def test_faithfulness_and_entailment():
    context = [
        "Product quantisation partitions high-dimensional vectors into orthogonal subspaces."
    ]
    faithful_answer = "Product quantisation partitions high-dimensional vectors into orthogonal subspaces. [doc:0]"
    unfaithful_answer = "The moon is made of green cheese and rabbits fly across mars."

    faith_high = calculate_faithfulness(faithful_answer, context)
    faith_low = calculate_faithfulness(unfaithful_answer, context)

    assert faith_high >= 0.80
    assert faith_low == 0.0

    # Claim entailment
    entail_high = evaluate_claim_entailment(faithful_answer, context)
    assert entail_high.entailed_claim_rate >= 0.50
    assert entail_high.unsupported_claim_rate <= 0.50

    entail_low = evaluate_claim_entailment(unfaithful_answer, context)
    assert entail_low.unsupported_claim_rate == 1.0
