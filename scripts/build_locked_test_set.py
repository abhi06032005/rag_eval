import pandas as pd

from ragbench.qa.labels import EvidenceSpan, QARecord
from ragbench.qa.validate import validate_qa_file

df = pd.read_parquet("data/processed/documents.parquet")
records = []

# Select 50 documents evenly across sources
arxiv_docs = df[df["source"] == "arxiv"].head(20)
wiki_docs = df[df["source"] == "wikipedia"].head(20)
gov_docs = df[df["source"] == "govinfo"].head(10)

selected = pd.concat([arxiv_docs, wiki_docs, gov_docs]).reset_index(drop=True)

for i, row in selected.iterrows():
    doc_id = row["doc_id"]
    text = row["text"]
    title = row["title"]
    source = row["source"]

    # Pick a specific sentence from text
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 30]
    candidate = lines[-1] if len(lines) > 1 else lines[0]

    start_c = text.find(candidate)
    end_c = start_c + len(candidate)

    if source == "arxiv":
        q = f"According to study '{title}', what finding or methodology is established?"
    elif source == "wikipedia":
        q = f"In the context of '{title}', what factual principle is described?"
    else:
        q = f"Under the legal or regulatory mandate of '{title}', what provision is stated?"

    record = QARecord(
        qa_id=f"qa-test-{i + 1:04d}",
        split="test",
        question=q,
        ground_truth_answer=candidate,
        answer_aliases=[candidate.lower(), candidate.rstrip(".")],
        gold_doc_ids=[doc_id],
        evidence_spans=[
            EvidenceSpan(
                doc_id=doc_id,
                start_char=start_c,
                end_char=end_c,
                text=candidate,
            )
        ],
        question_type="factoid",
        source_domain=source,
        creation_method="synthetic_template",
        review_status="unverified_programmatic",
    )
    records.append(record)

with open("data/qa/test.jsonl", "w", encoding="utf-8") as f:
    f.writelines(r.model_dump_json() + "\n" for r in records)

print(f"Wrote {len(records)} test records to data/qa/test.jsonl")
passed, rejected = validate_qa_file("data/qa/test.jsonl", "data/processed/documents.parquet")
print(f"Test set validation: {passed} passed, {rejected} rejected.")
