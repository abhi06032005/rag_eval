"""
Staged Experiment Grid Generator and Score Threshold Calibration.
Avoids uncontrolled Cartesian explosion by using four deliberate stages:
Representation -> Index/ANN -> Generation -> Load Saturation.
"""

from collections.abc import Sequence

import numpy as np

from ragbench.config import BenchmarkConfig, GridExperimentsConfig


def calibrate_dev_q05_gold(scores_of_relevant_hits: Sequence[float]) -> float:
    """Calculate 5th percentile of similarity scores of gold-relevant hits on dev set."""
    if not scores_of_relevant_hits:
        return 0.50
    return float(np.percentile(scores_of_relevant_hits, 5))


def calibrate_dev_max_f1(hit_scores: Sequence[tuple[float, bool]]) -> float:
    """
    Find similarity score threshold that maximizes F1 score on dev hits.
    hit_scores: list of (score, is_relevant)
    """
    if not hit_scores:
        return 0.50

    thresholds = sorted(list({round(s, 3) for s, _ in hit_scores}))
    best_thresh = 0.50
    best_f1 = -1.0

    for thresh in thresholds:
        tp = sum(1 for s, rel in hit_scores if s >= thresh and rel)
        fp = sum(1 for s, rel in hit_scores if s >= thresh and not rel)
        fn = sum(1 for s, rel in hit_scores if s < thresh and rel)

        prec = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return float(best_thresh)


def generate_representation_grid(grid_cfg: GridExperimentsConfig) -> list[BenchmarkConfig]:
    """Stage 1: 3 Embedding Models x 3 Chunk Sizes x 2 Overlaps = 18 configurations."""
    configs: list[BenchmarkConfig] = []
    rep = grid_cfg.representation
    for model in rep.embedding_models:
        dim = 1024 if "bge-m3" in model else 384
        for chunk_sz in rep.chunk_size_tokens:
            for overlap in rep.overlap_tokens:
                if overlap >= chunk_sz:
                    continue
                cfg = BenchmarkConfig(
                    embedding_model=model,
                    embedding_dimensions=dim,
                    chunk_size_tokens=chunk_sz,
                    overlap_tokens=overlap,
                    vector_db="qdrant",
                    hnsw_m=16,
                    ef_construct=128,
                    hnsw_ef=64,
                    quantization="none",
                    k=10,
                    score_threshold="none",
                )
                configs.append(cfg)
    return configs


def generate_index_grid(
    base_cfg: BenchmarkConfig, grid_cfg: GridExperimentsConfig
) -> list[BenchmarkConfig]:
    """
    Stage 2: Sweep HNSW m x ef_construct x quantisation x query hnsw_ef x k x thresholds.
    Inherits the best representation config from Stage 1.
    """
    configs: list[BenchmarkConfig] = []
    idx = grid_cfg.index
    for m in idx.hnsw_m:
        for ef_c in idx.ef_construct:
            for quant in idx.quantization:
                for hnsw_ef in idx.hnsw_ef:
                    for k in idx.k:
                        for thresh in idx.score_thresholds:
                            cfg = base_cfg.model_copy(
                                update={
                                    "hnsw_m": m,
                                    "ef_construct": ef_c,
                                    "quantization": quant,
                                    "hnsw_ef": hnsw_ef,
                                    "k": k,
                                    "score_threshold": thresh,
                                }
                            )
                            configs.append(cfg)
    return configs
