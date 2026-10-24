"""
QA Validation and Offset Verification.
Verifies that evidence spans match document text verbatim before freezing.
"""

import argparse
import json
import os
import sys

import pandas as pd

from ragbench.qa.labels import QARecord


def validate_qa_file(
    qa_path: str,
    documents_path: str = "data/processed/documents.parquet",
    output_frozen_path: str | None = None,
) -> tuple[int, int]:
    """
    Validate QA records against source document text.
    Returns (passed_count, rejected_count).
    """
    if not os.path.exists(documents_path):
        raise FileNotFoundError(f"Corpus documents not found at {documents_path}")

    docs_df = pd.read_parquet(documents_path)
    docs_by_id = {row["doc_id"]: row["text"] for _, row in docs_df.iterrows()}

    valid_records: list[QARecord] = []
    rejected_count = 0

    with open(qa_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            data = json.loads(line)
            record = QARecord(**data)

            # Verification checks
            is_valid = True
            for span in record.evidence_spans:
                if span.doc_id not in docs_by_id:
                    print(
                        f"Line {line_num}: Doc ID {span.doc_id} not found in corpus.",
                        file=sys.stderr,
                    )
                    is_valid = False
                    break

                doc_text = docs_by_id[span.doc_id]
                span_text = doc_text[span.start_char : span.end_char]
                if span.text and span.text.strip() != span_text.strip():
                    print(
                        f"Line {line_num}: Span text mismatch at [{span.start_char}:{span.end_char}].",
                        file=sys.stderr,
                    )
                    is_valid = False
                    break

            if is_valid:
                valid_records.append(record)
            else:
                rejected_count += 1

    if output_frozen_path:
        os.makedirs(os.path.dirname(output_frozen_path), exist_ok=True)
        with open(output_frozen_path, "w", encoding="utf-8") as f:
            f.writelines(r.model_dump_json() + "\n" for r in valid_records)
        print(f"Frozen {len(valid_records)} validated QA records to {output_frozen_path}")

    return len(valid_records), rejected_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and freeze QA dataset offsets.")
    parser.add_argument("qa_path", type=str, help="Input QA JSONL file")
    parser.add_argument(
        "--documents", type=str, default="data/processed/documents.parquet", help="Corpus Parquet"
    )
    parser.add_argument("--output", type=str, default=None, help="Frozen output JSONL path")
    args = parser.parse_args()

    passed, rejected = validate_qa_file(
        qa_path=args.qa_path,
        documents_path=args.documents,
        output_frozen_path=args.output,
    )
    print(f"Validation summary: {passed} passed, {rejected} rejected.")
    if rejected > 0 and not args.output:
        sys.exit(1)


if __name__ == "__main__":
    main()
