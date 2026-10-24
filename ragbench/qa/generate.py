"""
Synthetic Development QA Generator.
Extracts questions, answers, and verbatim character offsets from corpus documents
with automated rejection gates.
"""

import argparse
import os
import re

import pandas as pd

from ragbench.ids import make_qa_id
from ragbench.qa.labels import EvidenceSpan, QARecord


def generate_qa_from_document(doc_record: dict, idx: int) -> QARecord | None:
    """Extract a valid factoid QA item from a canonical document."""
    doc_id = doc_record["doc_id"]
    text = doc_record["text"]
    title = doc_record.get("title", "")
    source = doc_record.get("source", "generic")

    # Split into clean paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]
    if not paragraphs:
        paragraphs = [text.strip()]

    target_para = paragraphs[0] if len(paragraphs) == 1 else paragraphs[1]

    # Find candidate factual sentence
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", target_para) if len(s.strip()) > 30]
    if not sentences:
        return None

    fact_sentence = sentences[0]
    # Locate exact character offsets in full text
    start_char = text.find(fact_sentence)
    if start_char == -1:
        return None
    end_char = start_char + len(fact_sentence)

    # Automated rejection gate: maximum evidence span 500 chars
    if (end_char - start_char) > 500:
        return None

    # Synthesize question based on source domain
    if "quantisation" in fact_sentence.lower():
        question = "What memory and recall trade-offs occur during vector quantisation?"
        answer = fact_sentence
    elif "distributed" in fact_sentence.lower():
        question = "What are the primary operational challenges in distributed computing systems?"
        answer = fact_sentence
    elif "arxiv" in source.lower():
        question = f"What research question does the scientific abstract for '{title}' address?"
        answer = fact_sentence
    elif "govinfo" in source.lower():
        question = f"What statutory guideline or mandate is articulated in {title}?"
        answer = fact_sentence
    else:
        question = f"According to '{title}', what factual principle is established?"
        answer = fact_sentence

    qa_id = make_qa_id(idx)
    return QARecord(
        qa_id=qa_id,
        split="dev",
        question=question,
        ground_truth_answer=answer,
        answer_aliases=[answer.lower(), answer.rstrip(".")],
        gold_doc_ids=[doc_id],
        evidence_spans=[
            EvidenceSpan(
                doc_id=doc_id,
                start_char=start_char,
                end_char=end_char,
                text=fact_sentence,
            )
        ],
        question_type="factoid",
        source_domain=source,
        creation_method="synthetic",
        review_status="approved",
    )


def generate_dev_qa_set(
    documents_path: str = "data/processed/documents.parquet",
    output_path: str = "data/qa/dev.generated.jsonl",
    target_count: int = 100,
) -> int:
    """Generate target count of validated development QA records."""
    if not os.path.exists(documents_path):
        raise FileNotFoundError(f"Corpus file not found: {documents_path}. Run prepare first.")

    df = pd.read_parquet(documents_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    qa_records: list[QARecord] = []
    seen_questions: set[str] = set()

    for idx, row in df.iterrows():
        qa_item = generate_qa_from_document(row.to_dict(), len(qa_records) + 1)
        if qa_item is None:
            continue

        q_norm = qa_item.question.lower().strip()
        if q_norm in seen_questions:
            continue

        seen_questions.add(q_norm)
        qa_records.append(qa_item)

        if len(qa_records) >= target_count:
            break

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(q.model_dump_json() + "\n" for q in qa_records)

    print(f"Generated {len(qa_records)} validated QA records into {output_path}")
    return len(qa_records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic dev QA records.")
    parser.add_argument("--count", type=int, default=100, help="Number of questions to generate")
    parser.add_argument(
        "--documents", type=str, default="data/processed/documents.parquet", help="Path to corpus"
    )
    parser.add_argument(
        "--config", type=str, default="configs/qa_generation.yaml", help="QA generation config"
    )
    parser.add_argument(
        "--output", type=str, default="data/qa/dev.generated.jsonl", help="Output JSONL path"
    )
    args = parser.parse_args()

    generate_dev_qa_set(
        documents_path=args.documents,
        output_path=args.output,
        target_count=args.count,
    )


if __name__ == "__main__":
    main()
