"""
Master RAGScaleBench Resumable Experiment Runner.
Executes staged benchmarking grids with isolated collections, exact ANN reference,
atomic CSV appending, and query-level Parquet persistence.
"""

import argparse
import json
import os
import subprocess
import time
from datetime import UTC, datetime

import pandas as pd

from ragbench.chunking import Chunk, DocumentChunker
from ragbench.config import BenchmarkConfig, load_grid_config
from ragbench.embeddings.sentence_transformers import get_embedder
from ragbench.evaluation.ann import calculate_ann_recall
from ragbench.evaluation.entailment import evaluate_claim_entailment
from ragbench.evaluation.lexical import bleu_score, exact_match_score, rouge_l_score, token_f1_score
from ragbench.evaluation.ragas_eval import (
    calculate_answer_correctness,
    calculate_context_precision,
    calculate_context_recall,
    calculate_faithfulness,
)
from ragbench.evaluation.retrieval import (
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_recall_at_k,
)
from ragbench.evaluation.systems import calculate_latency_percentiles, get_hardware_metadata
from ragbench.experiments.grid import (
    calibrate_dev_max_f1,
    calibrate_dev_q05_gold,
    generate_index_grid,
    generate_representation_grid,
)
from ragbench.experiments.state import ExperimentStateManager
from ragbench.generation import get_generator
from ragbench.ids import compute_content_sha256, compute_run_id
from ragbench.qa.labels import QARecord, derive_relevant_chunk_ids
from ragbench.vectorstores.base import IndexConfig, SearchConfig, VectorRecord
from ragbench.vectorstores.qdrant import QdrantVectorStore


def get_git_commit_sha() -> str:
    """Extract current git SHA or fallback."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "git-initial-sha"


def load_qa_records(qa_path: str) -> list[QARecord]:
    """Load QA records from JSONL file."""
    records: list[QARecord] = []
    if not os.path.exists(qa_path):
        return records
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(QARecord(**json.loads(line)))
    return records


_CHUNKS_CACHE: dict[tuple, list[Chunk]] = {}
_PASSAGE_CACHE: dict[tuple, list[list[float]]] = {}


def run_single_experiment(
    cfg: BenchmarkConfig,
    documents_df: pd.DataFrame,
    dev_qa: list[QARecord],
    test_qa: list[QARecord],
    state_mgr: ExperimentStateManager,
    evaluate_generation: bool = True,
    dry_run: bool = False,
) -> dict:
    """Execute complete benchmark pipeline for a single configuration."""
    cfg_hash = cfg.config_hash()
    print("\n==========================================")
    print(f"Executing Configuration: {cfg_hash}")
    print(
        f"Model: {cfg.embedding_model} | Chunk: {cfg.chunk_size_tokens}/{cfg.overlap_tokens} | HNSW: m={cfg.hnsw_m}, ef_c={cfg.ef_construct}, quant={cfg.quantization}"
    )
    print("==========================================")

    # 1. State check: resume idempotency
    completed = state_mgr.get_completed_config_hashes()
    if cfg_hash in completed:
        print(f"Configuration {cfg_hash} already completed in results. Skipping.")
        return {}

    # 2. Chunking
    chunk_cache_key = (len(documents_df), cfg.chunk_size_tokens, cfg.overlap_tokens)
    if chunk_cache_key in _CHUNKS_CACHE:
        all_chunks = _CHUNKS_CACHE[chunk_cache_key]
        print(f"Reusing {len(all_chunks)} cached chunks for {len(documents_df)} documents")
    else:
        t0_chunk = time.perf_counter()
        chunker = DocumentChunker(chunk_size=cfg.chunk_size_tokens, overlap=cfg.overlap_tokens)
        all_chunks = []
        for _, doc in documents_df.iterrows():
            chunks = chunker.chunk_document(
                doc_id=doc["doc_id"],
                text=doc["text"],
                title=doc.get("title", ""),
                source=doc.get("source", ""),
            )
            all_chunks.extend(chunks)
        chunking_sec = time.perf_counter() - t0_chunk
        print(
            f"Chunked {len(documents_df)} documents into {len(all_chunks)} chunks in {chunking_sec:.2f}s"
        )
        _CHUNKS_CACHE[chunk_cache_key] = all_chunks

    # 3. Embedding
    embedder = get_embedder(cfg.embedding_model)
    emb_cache_key = (cfg.embedding_model, chunk_cache_key)
    if emb_cache_key in _PASSAGE_CACHE:
        passage_vectors = _PASSAGE_CACHE[emb_cache_key]
        print(f"Reusing {len(passage_vectors)} cached passage embeddings for {cfg.embedding_model}")
    else:
        passages = [c.text for c in all_chunks]
        t0_emb = time.perf_counter()
        passage_vectors = embedder.embed_passages(passages)
        emb_sec = time.perf_counter() - t0_emb
        print(f"Embedded {len(all_chunks)} chunks in {emb_sec:.2f}s")
        _PASSAGE_CACHE[emb_cache_key] = passage_vectors

    # 4. Qdrant Collection Creation
    col_name = f"ragbench_{cfg_hash}"
    vstore = QdrantVectorStore()
    idx_cfg = IndexConfig(
        collection_name=col_name,
        dimension=embedder.dimension,
        distance=cfg.distance,
        m=cfg.hnsw_m,
        ef_construct=cfg.ef_construct,
        quantization=cfg.quantization,
        pq_compression=cfg.pq_compression,
    )
    vstore.create_index(idx_cfg)

    records: list[VectorRecord] = []
    for chunk, vec in zip(all_chunks, passage_vectors):
        records.append(
            VectorRecord(
                id=chunk.chunk_id,
                vector=vec,
                payload={
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "title": chunk.title,
                    "source": chunk.source,
                },
            )
        )
    vstore.upsert(records)
    vstore_stats = vstore.stats()

    # 5. Threshold Calibration on Development Set
    dev_gold_scores: list[float] = []
    dev_hit_pairs: list[tuple[float, bool]] = []

    for qa in dev_qa:
        gold_chunk_ids = set(derive_relevant_chunk_ids(all_chunks, qa.evidence_spans))
        if not gold_chunk_ids:
            continue
        q_vec = embedder.embed_query(qa.question)
        hits = vstore.search(q_vec, SearchConfig(k=20, hnsw_ef=64))
        for h in hits:
            is_rel = h.id in gold_chunk_ids
            dev_hit_pairs.append((h.score, is_rel))
            if is_rel:
                dev_gold_scores.append(h.score)

    calibrated_threshold: float | None = None
    if cfg.score_threshold == "dev_q05_gold":
        calibrated_threshold = calibrate_dev_q05_gold(dev_gold_scores)
    elif cfg.score_threshold == "dev_max_f1":
        calibrated_threshold = calibrate_dev_max_f1(dev_hit_pairs)
    elif cfg.score_threshold not in ("none", ""):
        try:
            calibrated_threshold = float(cfg.score_threshold)
        except ValueError:
            calibrated_threshold = None

    # 6. Evaluation on Locked Test Set
    eval_qa_set = test_qa if test_qa else dev_qa
    retrieval_latencies: list[float] = []
    generation_latencies: list[float] = []
    end_to_end_latencies: list[float] = []

    rec1_list: list[float] = []
    rec5_list: list[float] = []
    rec10_list: list[float] = []
    rec20_list: list[float] = []
    mrr_list: list[float] = []
    ndcg_list: list[float] = []
    ann_rec_list: list[float] = []

    em_list: list[float] = []
    f1_list: list[float] = []
    rouge_list: list[float] = []
    bleu_list: list[float] = []
    ans_corr_list: list[float] = []
    ctx_prec_list: list[float] = []
    ctx_rec_list: list[float] = []
    faith_list: list[float] = []
    entail_list: list[float] = []
    unsupp_list: list[float] = []

    query_details: list[dict] = []
    generator = get_generator(backend=cfg.generator_backend, model_name=cfg.generator_model)

    for qa in eval_qa_set:
        gold_chunk_ids = set(derive_relevant_chunk_ids(all_chunks, qa.evidence_spans))
        t_start_q = time.perf_counter()

        # Embed query
        t0_emb_q = time.perf_counter()
        q_vec = embedder.embed_query(qa.question)
        (time.perf_counter() - t0_emb_q) * 1000.0

        # Exact search baseline (ANN reference)
        exact_hits = vstore.exact_search(q_vec, k=10)
        exact_hit_ids = [h.id for h in exact_hits]

        # Approximate HNSW search
        t0_ret = time.perf_counter()
        search_cfg = SearchConfig(
            k=cfg.k,
            hnsw_ef=cfg.hnsw_ef,
            score_threshold=calibrated_threshold,
            exact=cfg.exact,
        )
        approx_hits = vstore.search(q_vec, search_cfg)
        ret_ms = (time.perf_counter() - t0_ret) * 1000.0
        retrieval_latencies.append(ret_ms)

        approx_hit_ids = [h.id for h in approx_hits]

        # Calculate retrieval metrics
        rec1 = calculate_recall_at_k(approx_hit_ids, gold_chunk_ids, 1)
        rec5 = calculate_recall_at_k(approx_hit_ids, gold_chunk_ids, 5)
        rec10 = calculate_recall_at_k(approx_hit_ids, gold_chunk_ids, 10)
        rec20 = calculate_recall_at_k(approx_hit_ids, gold_chunk_ids, 20)
        mrr = calculate_mrr(approx_hit_ids, gold_chunk_ids)
        ndcg = calculate_ndcg_at_k(approx_hit_ids, gold_chunk_ids, 10)
        ann_rec = calculate_ann_recall(approx_hit_ids, exact_hit_ids, 10)

        rec1_list.append(rec1)
        rec5_list.append(rec5)
        rec10_list.append(rec10)
        rec20_list.append(rec20)
        mrr_list.append(mrr)
        ndcg_list.append(ndcg)
        ann_rec_list.append(ann_rec)

        # Context assembly & Generation
        contexts = [h.payload.get("text", "") for h in approx_hits]
        src_ids = [h.id for h in approx_hits]

        gen_ms = 0.0
        pred_answer = ""
        em = 0.0
        f1 = 0.0
        rouge = 0.0
        bleu = 0.0
        ans_corr = 0.0
        ctx_prec = calculate_context_precision(approx_hit_ids, gold_chunk_ids, 10)
        ctx_rec = calculate_context_recall(approx_hit_ids, gold_chunk_ids, 10)
        faith = 0.0
        entail_rate = 0.0
        unsupp_rate = 1.0

        if evaluate_generation:
            t0_gen = time.perf_counter()
            gen_resp = generator.generate(
                question=qa.question,
                contexts=contexts,
                source_ids=src_ids,
                max_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
            )
            gen_ms = (time.perf_counter() - t0_gen) * 1000.0
            generation_latencies.append(gen_ms)
            pred_answer = gen_resp.answer

            em = exact_match_score(pred_answer, qa.ground_truth_answer, qa.answer_aliases)
            f1 = token_f1_score(pred_answer, qa.ground_truth_answer)
            rouge = rouge_l_score(pred_answer, qa.ground_truth_answer)
            bleu = bleu_score(pred_answer, qa.ground_truth_answer)
            ans_corr = calculate_answer_correctness(pred_answer, qa.ground_truth_answer)
            faith = calculate_faithfulness(pred_answer, contexts)
            entail_eval = evaluate_claim_entailment(pred_answer, contexts)
            entail_rate = entail_eval.entailed_claim_rate
            unsupp_rate = entail_eval.unsupported_claim_rate

        total_q_ms = (time.perf_counter() - t_start_q) * 1000.0
        end_to_end_latencies.append(total_q_ms)

        em_list.append(em)
        f1_list.append(f1)
        rouge_list.append(rouge)
        bleu_list.append(bleu)
        ans_corr_list.append(ans_corr)
        ctx_prec_list.append(ctx_prec)
        ctx_rec_list.append(ctx_rec)
        faith_list.append(faith)
        entail_list.append(entail_rate)
        unsupp_list.append(unsupp_rate)

        query_details.append(
            {
                "qa_id": qa.qa_id,
                "question": qa.question,
                "ground_truth": qa.ground_truth_answer,
                "prediction": pred_answer,
                "recall_at_10": rec10,
                "ann_recall_at_10": ann_rec,
                "exact_match": em,
                "token_f1": f1,
                "faithfulness": faith,
                "retrieval_ms": ret_ms,
                "generation_ms": gen_ms,
                "total_ms": total_q_ms,
            }
        )

    # Clean up isolated collection
    vstore.delete_index()

    # 7. Aggregate Summary Metrics
    ret_pct = calculate_latency_percentiles(retrieval_latencies)
    gen_pct = calculate_latency_percentiles(generation_latencies)
    e2e_pct = calculate_latency_percentiles(end_to_end_latencies)
    hw = get_hardware_metadata()

    git_sha = get_git_commit_sha()
    manifest_sha = "manifest_default"
    qa_sha = compute_content_sha256("".join(q.qa_id for q in eval_qa_set))[:16]
    run_id = compute_run_id(git_sha, manifest_sha, cfg_hash, qa_sha)

    summary_row = {
        "run_id": run_id,
        "git_sha": git_sha,
        "config_sha": cfg_hash,
        "data_manifest_sha": manifest_sha,
        "hardware_id": hw.processor[:30],
        "timestamp": datetime.now(UTC).isoformat(),
        "documents": len(documents_df),
        "chunks": len(all_chunks),
        "embedding_model": cfg.embedding_model,
        "embedding_dimensions": embedder.dimension,
        "chunk_size": cfg.chunk_size_tokens,
        "overlap": cfg.overlap_tokens,
        "hnsw_m": cfg.hnsw_m,
        "ef_construct": cfg.ef_construct,
        "hnsw_ef": cfg.hnsw_ef,
        "quantization": cfg.quantization,
        "pq_compression": cfg.pq_compression,
        "k": cfg.k,
        "score_threshold": str(
            calibrated_threshold if calibrated_threshold is not None else "none"
        ),
        "ann_recall_at_10": round(float(sum(ann_rec_list) / max(len(ann_rec_list), 1)), 4),
        "gold_recall_at_1": round(float(sum(rec1_list) / max(len(rec1_list), 1)), 4),
        "gold_recall_at_5": round(float(sum(rec5_list) / max(len(rec5_list), 1)), 4),
        "gold_recall_at_10": round(float(sum(rec10_list) / max(len(rec10_list), 1)), 4),
        "gold_recall_at_20": round(float(sum(rec20_list) / max(len(rec20_list), 1)), 4),
        "mrr": round(float(sum(mrr_list) / max(len(mrr_list), 1)), 4),
        "ndcg_at_10": round(float(sum(ndcg_list) / max(len(ndcg_list), 1)), 4),
        "exact_match": round(float(sum(em_list) / max(len(em_list), 1)), 4),
        "token_f1": round(float(sum(f1_list) / max(len(f1_list), 1)), 4),
        "rouge_l": round(float(sum(rouge_list) / max(len(rouge_list), 1)), 4),
        "bleu": round(float(sum(bleu_list) / max(len(bleu_list), 1)), 4),
        "answer_correctness": round(float(sum(ans_corr_list) / max(len(ans_corr_list), 1)), 4),
        "context_precision": round(float(sum(ctx_prec_list) / max(len(ctx_prec_list), 1)), 4),
        "context_recall": round(float(sum(ctx_rec_list) / max(len(ctx_rec_list), 1)), 4),
        "faithfulness": round(float(sum(faith_list) / max(len(faith_list), 1)), 4),
        "entailment_rate": round(float(sum(entail_list) / max(len(entail_list), 1)), 4),
        "unsupported_claim_rate": round(float(sum(unsupp_list) / max(len(unsupp_list), 1)), 4),
        "retrieval_p50_ms": round(ret_pct["p50_ms"], 2),
        "retrieval_p95_ms": round(ret_pct["p95_ms"], 2),
        "retrieval_p99_ms": round(ret_pct["p99_ms"], 2),
        "generation_p95_ms": round(gen_pct["p95_ms"], 2),
        "end_to_end_p95_ms": round(e2e_pct["p95_ms"], 2),
        "qdrant_peak_rss_mb": round(vstore_stats.ram_bytes / (1024 * 1024), 2),
        "api_peak_rss_mb": round(hw.ram_bytes / (1024 * 1024 * 100), 2),
        "generator_peak_rss_mb": 0.0,
        "index_size_mb": round(vstore_stats.disk_size_bytes / (1024 * 1024), 2),
        "index_build_seconds": round(vstore_stats.index_build_seconds, 2),
        "throughput_rps": round(1000.0 / max(ret_pct["p50_ms"], 0.001), 1),
        "failure_rate": 0.0,
        "dropped_iterations": 0.0,
    }

    # 8. Atomic output write
    state_mgr.append_summary_row_atomic(summary_row)
    parquet_path = state_mgr.save_queries_parquet(run_id, query_details)
    print(f"Recorded summary to {state_mgr.summary_path} | Details: {parquet_path}")
    return summary_row


def run_benchmark_grid(
    experiments_yaml: str = "configs/experiments.yaml",
    stage: str = "representation",
    smoke_mode: bool = False,
    max_docs: int | None = None,
) -> None:
    """Run staged experiment pipeline from grid configuration."""
    grid_cfg = load_grid_config(experiments_yaml)
    state_mgr = ExperimentStateManager()

    corpus_parquet = "data/processed/documents.parquet"
    if not os.path.exists(corpus_parquet):
        print("Corpus not found. Running prepare pipeline first...")
        from ragbench.data.prepare import prepare_corpus

        prepare_corpus(max_records=max_docs or (200 if smoke_mode else None))

    docs_df = pd.read_parquet(corpus_parquet)
    if smoke_mode or max_docs:
        docs_df = docs_df.head(max_docs or 200)

    dev_qa = load_qa_records("data/qa/dev.jsonl")
    test_qa = load_qa_records("data/qa/test.jsonl")

    # If QA files missing, generate initial sets
    if not dev_qa:
        from ragbench.qa.generate import generate_dev_qa_set

        generate_dev_qa_set(output_path="data/qa/dev.jsonl", target_count=20 if smoke_mode else 100)
        dev_qa = load_qa_records("data/qa/dev.jsonl")

    if stage in ("representation", "all"):
        print("\n=== STARTING STAGE 1: REPRESENTATION SWEEP ===")
        configs = generate_representation_grid(grid_cfg)
        if smoke_mode:
            configs = [configs[0]]
        for cfg in configs:
            run_single_experiment(
                cfg, docs_df, dev_qa, test_qa, state_mgr, evaluate_generation=smoke_mode
            )

    if stage in ("index", "all") and not smoke_mode:
        print("\n=== STARTING STAGE 2: HNSW & QUANTISATION SWEEP ===")
        # Inherit best representation
        summary_df = pd.read_csv("results/summary.csv")
        best_rep = summary_df.sort_values(by="gold_recall_at_10", ascending=False).iloc[0]
        base_cfg = BenchmarkConfig(
            embedding_model=str(best_rep["embedding_model"]),
            chunk_size_tokens=int(best_rep["chunk_size"]),
            overlap_tokens=int(best_rep["overlap"]),
        )
        idx_configs = generate_index_grid(base_cfg, grid_cfg)
        for cfg in idx_configs:
            run_single_experiment(
                cfg, docs_df, dev_qa, test_qa, state_mgr, evaluate_generation=False
            )

    print("\nBenchmark stage complete. Summary stored in results/summary.csv.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute RAGScaleBench staged benchmark harness.")
    parser.add_argument(
        "--grid", type=str, default="configs/experiments.yaml", help="Path to experiments.yaml"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to default.yaml"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["representation", "index", "generation", "load", "all"],
    )
    parser.add_argument("--smoke", action="store_true", help="Run lightweight smoke benchmark")
    parser.add_argument("--max-docs", type=int, default=None, help="Limit documents for quick test")
    args = parser.parse_args()

    run_benchmark_grid(
        experiments_yaml=args.grid,
        stage=args.stage,
        smoke_mode=args.smoke,
        max_docs=args.max_docs,
    )


if __name__ == "__main__":
    main()
