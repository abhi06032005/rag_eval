import json
import os
import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse

from app.dashboard_api import (
    get_available_runs,
    get_benchmark_summary_data,
    get_corpus_statistics,
    get_load_test_data,
    get_qdrant_audit_data,
    get_run_queries,
    get_scifact_benchmark_data,
    get_scientific_integrity_audit_data,
    get_generation_evaluation_data,
)

from app.dependencies import get_generator_instance, get_pipeline_components
from app.schemas import (
    AskRequest,
    AskResponse,
    RetrieveRequest,
    RetrieveResponse,
    SourceRecord,
    TimingBreakdown,
)
from app.telemetry import TelemetryMiddleware, global_metrics
from ragbench.generation import (
    OllamaConnectionError,
    OpenAIConnectionError,
    is_abstention_answer,
    list_generator_providers,
)
from ragbench.vectorstores.base import SearchConfig

logger = structlog.get_logger()



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure pipeline components initialized
    components = get_pipeline_components()
    logger.info(
        "RAGScaleBench service starting up",
        embedding_model=components.embedder.model_name,
        vector_db=components.config.vector_db,
        generator_model=components.generator.model_name,
    )
    yield
    logger.info("RAGScaleBench service shutting down gracefully")


app = FastAPI(
    title="RAGScaleBench Service",
    description="Production-grade RAG benchmarking and serving engine",
    version="0.1.0",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(TelemetryMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/healthz", tags=["Operations"])
async def healthz():
    """Liveness probe."""
    return {"status": "ok", "service": "ragscalebench", "timestamp": time.time()}


@app.get("/readyz", tags=["Operations"])
async def readyz():
    """Readiness probe checking VectorStore and Embedder."""
    components = get_pipeline_components()
    try:
        stats = components.vector_store.stats()
        return {
            "status": "ready",
            "vector_store": components.config.vector_db,
            "active_collection": stats.collection_name,
            "total_vectors": stats.total_vectors,
            "embedder": components.embedder.model_name,
            "dimension": components.embedder.dimension,
            "generator": components.generator.model_name,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Service not ready: {exc}")


@app.get("/metrics", response_class=PlainTextResponse, tags=["Operations"])
async def metrics():
    """Prometheus telemetry scrape endpoint."""
    return global_metrics.format_prometheus()


@app.post("/retrieve", response_model=RetrieveResponse, tags=["Retrieval"])
async def retrieve(req: RetrieveRequest, request: Request):
    """
    Retrieve nearest neighbours without generation.
    Allows independent vector search latency and quality benchmarking.
    """
    req_id = request.headers.get("X-Request-ID", f"req-{uuid.uuid4().hex[:12]}")
    components = get_pipeline_components()
    t_start = time.perf_counter()

    # 1. Embedding
    t0 = time.perf_counter()
    query_vector = components.embedder.embed_query(req.query)
    emb_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Retrieval
    t1 = time.perf_counter()
    search_cfg = SearchConfig(
        k=req.k,
        hnsw_ef=req.hnsw_ef,
        score_threshold=req.score_threshold,
        exact=req.exact,
    )
    hits = components.vector_store.search(vector=query_vector, config=search_cfg)
    ret_ms = (time.perf_counter() - t1) * 1000.0
    total_ms = (time.perf_counter() - t_start) * 1000.0

    source_records = [
        SourceRecord(
            chunk_id=h.id,
            doc_id=h.payload.get("doc_id", "unknown"),
            score=h.score,
            text=h.payload.get("text", ""),
            title=h.payload.get("title", ""),
            source=h.payload.get("source", ""),
        )
        for h in hits
    ]

    return RetrieveResponse(
        request_id=req_id,
        query=req.query,
        hits=source_records,
        timings_ms=TimingBreakdown(
            embedding=emb_ms,
            retrieval=ret_ms,
            context_assembly=0.0,
            generation=0.0,
            total=total_ms,
        ),
        config_hash=components.config.config_hash(),
    )


@app.post("/ask", response_model=AskResponse, tags=["RAG Pipeline"])
async def ask(req: AskRequest, request: Request):
    """
    Full RAG pipeline: Query -> Embedding -> Retrieval -> Delimited Context -> LLM Generation.
    Strictly verifies LLM connectivity; raises HTTP 503 instead of silent mock fallback when real mode is requested.
    """
    req_id = request.headers.get("X-Request-ID", f"req-{uuid.uuid4().hex[:12]}")
    components = get_pipeline_components()
    t_start = time.perf_counter()

    # 1. Embedding
    t0 = time.perf_counter()
    query_vector = components.embedder.embed_query(req.question)
    emb_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Retrieval
    t1 = time.perf_counter()
    search_cfg = SearchConfig(
        k=req.k,
        hnsw_ef=req.hnsw_ef,
        score_threshold=req.score_threshold,
    )
    hits = components.vector_store.search(vector=query_vector, config=search_cfg)
    ret_ms = (time.perf_counter() - t1) * 1000.0

    # 3. Context assembly
    t2 = time.perf_counter()
    contexts = [h.payload.get("text", "") for h in hits]
    source_ids = [f"{h.payload.get('doc_id', 'doc')}:{h.id}" for h in hits]
    ctx_ms = (time.perf_counter() - t2) * 1000.0

    # 4. Generation
    t3 = time.perf_counter()
    generator = get_generator_instance(req.generator_backend, req.model)
    try:
        gen_resp = generator.generate(
            question=req.question,
            contexts=contexts,
            source_ids=source_ids,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
    except (OllamaConnectionError, OpenAIConnectionError) as exc:
        logger.error("LLM generation failed", backend=req.generator_backend, error=str(exc))
        raise HTTPException(
            status_code=503,
            detail=f"Real generation backend unavailable: {exc}. "
                   "Silent mock fallback is strictly disabled in real mode."
        )
    except Exception as exc:
        logger.error("Unexpected generation error", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")

    gen_ms = (time.perf_counter() - t3) * 1000.0
    total_ms = (time.perf_counter() - t_start) * 1000.0

    source_records = [
        SourceRecord(
            chunk_id=h.id,
            doc_id=h.payload.get("doc_id", "unknown"),
            score=h.score,
            text=h.payload.get("text", ""),
            title=h.payload.get("title", ""),
            source=h.payload.get("source", ""),
        )
        for h in hits
    ]

    generator_info = {
        "backend": gen_resp.backend,
        "model_name": gen_resp.model_name,
        "runtime_mode": "mock_fixture" if gen_resp.is_mock else "real_llm",
        "is_mock": gen_resp.is_mock,
        "token_count": gen_resp.token_count,
        "prompt_tokens": gen_resp.prompt_tokens,
        "completion_tokens": gen_resp.completion_tokens,
        "generation_ms": gen_resp.duration_ms,
        "abstained": gen_resp.abstained,
        "finish_reason": gen_resp.finish_reason,
    }

    return AskResponse(
        request_id=req_id,
        question=req.question,
        answer=gen_resp.answer,
        sources=source_ids,
        source_details=source_records,
        timings_ms=TimingBreakdown(
            embedding=emb_ms,
            retrieval=ret_ms,
            context_assembly=ctx_ms,
            generation=gen_ms,
            total=total_ms,
        ),
        config_hash=components.config.config_hash(),
        abstained=gen_resp.abstained,
        generator_info=generator_info,
    )


@app.post("/ask/stream", tags=["RAG Pipeline"])
async def ask_stream(req: AskRequest, request: Request):
    """
    Server-Sent Events (SSE) streaming endpoint for RAG generation.
    Emits token events as they arrive, and a terminal event with citations, metrics, and metadata.
    """
    req_id = request.headers.get("X-Request-ID", f"req-{uuid.uuid4().hex[:12]}")
    components = get_pipeline_components()
    t_start = time.perf_counter()

    # 1. Embedding
    t0 = time.perf_counter()
    query_vector = components.embedder.embed_query(req.question)
    emb_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Retrieval
    t1 = time.perf_counter()
    search_cfg = SearchConfig(
        k=req.k,
        hnsw_ef=req.hnsw_ef,
        score_threshold=req.score_threshold,
    )
    hits = components.vector_store.search(vector=query_vector, config=search_cfg)
    ret_ms = (time.perf_counter() - t1) * 1000.0

    # 3. Context assembly
    t2 = time.perf_counter()
    contexts = [h.payload.get("text", "") for h in hits]
    source_ids = [f"{h.payload.get('doc_id', 'doc')}:{h.id}" for h in hits]
    ctx_ms = (time.perf_counter() - t2) * 1000.0

    source_records = [
        SourceRecord(
            chunk_id=h.id,
            doc_id=h.payload.get("doc_id", "unknown"),
            score=h.score,
            text=h.payload.get("text", ""),
            title=h.payload.get("title", ""),
            source=h.payload.get("source", ""),
        )
        for h in hits
    ]

    generator = get_generator_instance(req.generator_backend, req.model)

    def sse_event_stream():
        t_gen_start = time.perf_counter()
        token_accum = []
        try:
            for chunk in generator.generate_stream(
                question=req.question,
                contexts=contexts,
                source_ids=source_ids,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            ):
                token_accum.append(chunk.token)
                event_data = {
                    "event": "token",
                    "token": chunk.token,
                    "is_finished": False,
                    "token_count": chunk.token_count,
                }
                yield f"data: {json.dumps(event_data)}\n\n"

            full_answer = "".join(token_accum).strip()
            gen_ms = (time.perf_counter() - t_gen_start) * 1000.0
            total_ms = (time.perf_counter() - t_start) * 1000.0
            abstained = is_abstention_answer(full_answer)

            terminal_payload = {
                "event": "done",
                "is_finished": True,
                "request_id": req_id,
                "question": req.question,
                "answer": full_answer,
                "sources": source_ids,
                "source_details": [s.model_dump() for s in source_records],
                "timings_ms": {
                    "embedding": emb_ms,
                    "retrieval": ret_ms,
                    "context_assembly": ctx_ms,
                    "generation": gen_ms,
                    "total": total_ms,
                },
                "generator_info": {
                    "backend": generator.backend_name,
                    "model_name": generator.model_name,
                    "runtime_mode": "mock_fixture" if getattr(generator, "backend_name", "") == "mock" else "real_llm",
                    "is_mock": getattr(generator, "backend_name", "") == "mock",
                    "token_count": len(token_accum),
                    "generation_ms": gen_ms,
                    "abstained": abstained,
                },
                "abstained": abstained,
                "config_hash": components.config.config_hash(),
            }
            yield f"data: {json.dumps(terminal_payload)}\n\n"
        except (OllamaConnectionError, OpenAIConnectionError) as exc:
            err_payload = {
                "event": "error",
                "is_finished": True,
                "error": str(exc),
                "backend": generator.backend_name,
            }
            yield f"data: {json.dumps(err_payload)}\n\n"
        except Exception as exc:
            err_payload = {
                "event": "error",
                "is_finished": True,
                "error": f"Streaming error: {exc}",
            }
            yield f"data: {json.dumps(err_payload)}\n\n"

    return StreamingResponse(sse_event_stream(), media_type="text/event-stream")


@app.get("/api/generator/status", tags=["RAG Pipeline"])
async def generator_status():
    """Inspect availability, health, and configuration of all supported generator backends."""
    return list_generator_providers()


@app.get("/api/scientific-audit", tags=["Scientific Integrity"])
async def scientific_audit():
    """Return Priority 1 forensic scientific integrity audit results and split ground-truth."""
    return get_scientific_integrity_audit_data()


@app.get("/api/generation-eval", tags=["RAG Generation"])
async def generation_eval():
    """Return Priority 4 Generative Answer Evaluation results."""
    return get_generation_evaluation_data()




# Dashboard & Research Telemetry Endpoints
DASHBOARD_HTML_PATH = os.path.join(os.path.dirname(__file__), "templates", "dashboard.html")


@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
async def dashboard():
    """Serve interactive systems research dashboard."""
    if not os.path.exists(DASHBOARD_HTML_PATH):
        raise HTTPException(status_code=404, detail="Dashboard template not found")
    with open(DASHBOARD_HTML_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/benchmark/summary", tags=["Dashboard API"])
async def benchmark_summary():
    """Retrieve full experimental benchmark matrix with human-readable labels."""
    return get_benchmark_summary_data()


@app.get("/api/benchmark/download", tags=["Dashboard API"])
async def benchmark_download():
    """Download summary.csv experimental results file."""
    csv_path = "results/summary.csv"
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="No benchmark results found on disk")
    return FileResponse(csv_path, media_type="text/csv", filename="ragscalebench_summary.csv")


@app.get("/api/benchmark/runs", tags=["Dashboard API"])
async def benchmark_runs():
    """List all available detailed experiment run directories."""
    return get_available_runs()


@app.get("/api/benchmark/runs/{run_id}", tags=["Dashboard API"])
async def benchmark_run_details(run_id: str, limit: int = 50):
    """Fetch sample per-question query execution details for a given run."""
    return get_run_queries(run_id, limit=limit)


@app.get("/api/corpus/stats", tags=["Dashboard API"])
async def corpus_stats():
    """Fetch ingested corpus counts and verified source distributions."""
    return get_corpus_statistics()


@app.get("/api/benchmark/scifact", tags=["Dashboard API"])
async def benchmark_scifact():
    """Retrieve authentic BEIR SciFact benchmark results."""
    return get_scifact_benchmark_data()


@app.get("/api/benchmark/qdrant-audit", tags=["Dashboard API"])
async def benchmark_qdrant_audit():
    """Retrieve real Qdrant server audit metrics and on-disk memory telemetry."""
    return get_qdrant_audit_data()


@app.get("/api/benchmark/load-test", tags=["Dashboard API"])
async def benchmark_load_test():
    """Retrieve concurrency ladder saturation sweep results."""
    return get_load_test_data()


@app.get("/api/benchmark/scientific-integrity", tags=["Dashboard API"])
async def benchmark_scientific_integrity():
    """Retrieve scientific integrity and statistical audit data."""
    return get_scientific_integrity_audit_data()


@app.get("/api/benchmark/generation-evaluation", tags=["Dashboard API"])
async def benchmark_generation_evaluation():
    """Retrieve real RAG generative answer evaluation and metrics."""
    return get_generation_evaluation_data()


# Dashboard API Aliases
@app.get("/api/dashboard/summary", tags=["Dashboard API"])
async def dashboard_summary():
    return get_benchmark_summary_data()


@app.get("/api/dashboard/scientific-integrity", tags=["Dashboard API"])
async def dashboard_scientific_integrity():
    return get_scientific_integrity_audit_data()


@app.get("/api/dashboard/scifact", tags=["Dashboard API"])
async def dashboard_scifact():
    return get_scifact_benchmark_data()


@app.get("/api/dashboard/load-test", tags=["Dashboard API"])
async def dashboard_load_test():
    return get_load_test_data()


@app.get("/api/dashboard/qdrant-audit", tags=["Dashboard API"])
async def dashboard_qdrant_audit():
    return get_qdrant_audit_data()


@app.get("/api/dashboard/generation-evaluation", tags=["Dashboard API"])
async def dashboard_generation_eval():
    return get_generation_evaluation_data()


# Phase 5 Research Endpoints
from app.dashboard_api import (
    get_phase5_ablations,
    get_phase5_bootstrap_data,
    get_phase5_datasets_data,
    get_phase5_failure_cases,
    get_phase5_routing_data,
    get_phase5_service_perf,
)


@app.get("/api/phase5/datasets", tags=["Phase 5 Research"])
async def phase5_datasets():
    """Cross-domain benchmark dataset metrics (SciFact, NFCorpus, FiQA)."""
    return get_phase5_datasets_data()


@app.get("/api/phase5/bootstrap", tags=["Phase 5 Research"])
async def phase5_bootstrap(dataset: str = "scifact"):
    """Paired query bootstrap results and 95% confidence intervals."""
    return get_phase5_bootstrap_data(dataset=dataset)


@app.get("/api/phase5/routing", tags=["Phase 5 Research"])
async def phase5_routing(dataset: str = "scifact", limit: int = 50):
    """Adaptive routing decisions and query features."""
    return get_phase5_routing_data(dataset=dataset, limit=limit)


@app.get("/api/phase5/failures", tags=["Phase 5 Research"])
async def phase5_failures():
    """Forensic failure cases where Hybrid or Adaptive degraded."""
    return get_phase5_failure_cases()


@app.get("/api/phase5/ablations", tags=["Phase 5 Research"])
async def phase5_ablations():
    """All Stage 5 ablation study sweeps."""
    return get_phase5_ablations()


@app.get("/api/phase5/service-perf", tags=["Phase 5 Research"])
async def phase5_service_perf():
    """Stage 6 real service HTTP benchmark load test results."""
    return get_phase5_service_perf()


# Phase 6 Research Endpoints
from app.dashboard_api import (
    get_adaptive_runtime_audit_data,
    get_canonical_registry_data,
    get_generation_audit_data,
)


@app.get("/api/phase6/canonical-registry", tags=["Phase 6 Research"])
async def phase6_canonical_registry():
    """Phase 6 Canonical Results Registry with reproducible manifests."""
    return get_canonical_registry_data()


@app.get("/api/phase6/runtime-audit", tags=["Phase 6 Research"])
async def phase6_runtime_audit():
    """Phase 6 Task 3 Runtime Cost & Timing Breakdown Audit."""
    return get_adaptive_runtime_audit_data()


@app.get("/api/phase6/generation-audit", tags=["Phase 6 Research"])
async def phase6_generation_audit():
    """Phase 6 Task 6 Generation Pipeline Reality & Entailment Audit."""
    return get_generation_audit_data()


@app.get("/api/phase5/download/{artifact_name}", tags=["Phase 5 & 6 Research"])
async def phase5_download(artifact_name: str):
    """Download specific experimental results artifact."""
    artifact_map = {
        "scifact_corrected": "results/scifact/corrected_results.csv",
        "scifact_bootstrap": "results/scifact/bootstrap_comparisons.csv",
        "scifact_manifest": "results/scifact/audit_manifest.json",
        "scifact_routing": "results/scifact/adaptive_routing_decisions.jsonl",
        "nfcorpus_summary": "results/nfcorpus/metrics_summary.csv",
        "nfcorpus_bootstrap": "results/nfcorpus/bootstrap_comparisons.csv",
        "fiqa_summary": "results/fiqa/metrics_summary.csv",
        "canonical_registry_csv": "results/canonical_results_registry.csv",
        "canonical_registry_json": "results/canonical_results_registry.json",
        "adaptive_runtime_csv": "results/service_performance/adaptive_runtime_breakdown.csv",
        "adaptive_runtime_json": "results/service_performance/adaptive_runtime_audit.json",
        "generation_audit_json": "results/qa/generation_verification_audit.json",
        "service_performance": "results/service_performance/http_benchmark_summary.csv",
        "fusion_ablation": "results/ablations/fusion_parameter_ablation.csv",
        "topk_ablation": "results/ablations/topk_cutoff_ablation.csv",
        "feature_ablation": "results/ablations/adaptive_feature_ablation.csv",
        "quant_ablation": "results/ablations/quantization_rescore_ablation.csv",
        "scientific_integrity_audit": "results/scientific_integrity_audit.json",
        "scientific_integrity_report": "reports/scientific_integrity_audit.md",
        "generation_eval_json": "results/qa/generation_evaluation_results.json",
        "generation_eval_report": "reports/rag_generation_evaluation.md",
    }
    target = artifact_map.get(artifact_name)
    if not target or not os.path.exists(target):
        raise HTTPException(status_code=404, detail=f"Artifact {artifact_name} not found")
    media_type = "application/json" if target.endswith(".json") or target.endswith(".jsonl") else "text/csv"
    return FileResponse(target, media_type=media_type, filename=os.path.basename(target))


