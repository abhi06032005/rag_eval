import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_pipeline_components
from app.main import app
from ragbench.vectorstores.base import IndexConfig, VectorRecord


@pytest.fixture(scope="module")
def api_client():
    components = get_pipeline_components()
    # Seed in-memory vector store with sample chunk
    idx_cfg = IndexConfig(collection_name="api_test_col", dimension=components.embedder.dimension)
    components.vector_store.create_index(idx_cfg)
    v = components.embedder.embed_query("dense retrieval and vector quantisation")
    components.vector_store.upsert(
        [
            VectorRecord(
                id="seed_chunk_1",
                vector=v,
                payload={
                    "doc_id": "seed_doc_1",
                    "text": "Vector quantisation significantly compresses dense representations.",
                    "title": "Quantisation Overview",
                    "source": "arxiv",
                },
            )
        ]
    )
    client = TestClient(app)
    return client


def test_healthz_and_readyz(api_client):
    r1 = api_client.get("/healthz")
    assert r1.status_code == 200
    assert r1.json()["status"] == "ok"

    r2 = api_client.get("/readyz")
    assert r2.status_code == 200
    assert r2.json()["status"] == "ready"


def test_metrics_endpoint(api_client):
    r = api_client.get("/metrics")
    assert r.status_code == 200
    assert "ragscalebench_requests_total" in r.text


def test_retrieve_endpoint(api_client):
    payload = {
        "query": "vector quantisation compression",
        "k": 5,
        "hnsw_ef": 64,
    }
    r = api_client.post("/retrieve", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "request_id" in data
    assert "timings_ms" in data
    assert "retrieval" in data["timings_ms"]
    assert "embedding" in data["timings_ms"]
    assert len(data["hits"]) >= 1


def test_ask_endpoint(api_client):
    payload = {
        "question": "What does vector quantisation do?",
        "k": 3,
        "hnsw_ef": 64,
        "temperature": 0.0,
    }
    r = api_client.post("/ask", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "request_id" in data
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert "sources" in data
    assert "timings_ms" in data
    assert data["timings_ms"]["generation"] >= 0.0


def test_phase6_endpoints(api_client):
    r1 = api_client.get("/api/phase6/canonical-registry")
    assert r1.status_code == 200
    reg = r1.json()
    assert "records" in reg
    assert len(reg["records"]) >= 10
    assert "checksums" in reg
    assert "non_inferiority_analysis" in reg

    r2 = api_client.get("/api/phase6/runtime-audit")
    assert r2.status_code == 200
    runtime = r2.json()
    assert "summary_comparison" in runtime
    comp = {row["Metric / Component"]: row for row in runtime["summary_comparison"]}
    assert comp["Neural Embedding Invocations"]["Adaptive_Router"] == 162
    assert comp["Qdrant / Vector Search Requests"]["Adaptive_Router"] == 162


    r3 = api_client.get("/api/phase6/generation-audit")
    assert r3.status_code == 200
    gen = r3.json()
    assert "summary" in gen
    assert gen["summary"]["origin_classification"]["is_mock_generator"] is True
    assert gen["summary"]["origin_classification"]["is_actual_llm"] is False

    r4 = api_client.get("/api/phase5/download/canonical_registry_csv")
    assert r4.status_code == 200
    assert "text/csv" in r4.headers.get("content-type", "")

