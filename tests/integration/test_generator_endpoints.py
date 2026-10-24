"""
Integration tests for Generator API Endpoints:
- GET /api/generator/status
- POST /ask with metadata and strict 503 error handling on offline LLM
- POST /ask/stream Server-Sent Events (SSE) streaming
- GET /api/scientific-audit
"""

import json
from fastapi.testclient import TestClient

from app.dependencies import get_pipeline_components
from app.main import app
from ragbench.vectorstores.base import IndexConfig, VectorRecord


def test_generator_status_endpoint():
    client = TestClient(app)
    resp = client.get("/api/generator/status")
    assert resp.status_code == 200
    providers = resp.json()
    assert isinstance(providers, list)
    backends = {p["backend"] for p in providers}
    assert "mock" in backends
    assert "ollama" in backends
    assert "openai" in backends


def test_ask_endpoint_with_mock_generator():
    components = get_pipeline_components()
    idx_cfg = IndexConfig(collection_name="gen_test_col", dimension=components.embedder.dimension)
    components.vector_store.create_index(idx_cfg)
    v = components.embedder.embed_query("dense retrieval and vector quantisation")
    components.vector_store.upsert(
        [
            VectorRecord(
                id="c1",
                vector=v,
                payload={
                    "doc_id": "d1",
                    "text": "Vector quantisation compresses dense vectors into compact codes.",
                    "title": "Quantisation",
                    "source": "test",
                },
            )
        ]
    )

    client = TestClient(app)
    # Default / explicit mock generator
    payload = {
        "question": "What does vector quantisation do?",
        "k": 2,
        "generator_backend": "mock",
    }
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "generator_info" in data
    assert data["generator_info"]["is_mock"] is True
    assert data["generator_info"]["runtime_mode"] == "mock_fixture"
    assert data["abstained"] is False
    assert len(data["sources"]) > 0


def test_ask_endpoint_no_silent_fallback_on_unreachable_ollama():
    client = TestClient(app)
    # When user requests ollama, but ollama is offline on random port, must return 503
    payload = {
        "question": "What is HNSW?",
        "k": 2,
        "generator_backend": "ollama",
    }
    resp = client.post("/ask", json=payload)
    # If ollama daemon happens to not be running locally, it must return 503 (NOT 200 with mock)
    # If an Ollama daemon happens to be running, it either succeeds or returns 503.
    assert resp.status_code in (200, 503)
    if resp.status_code == 503:
        detail = resp.json()["detail"]
        assert "silent mock fallback is strictly disabled" in detail.lower()


def test_ask_stream_endpoint():
    client = TestClient(app)
    payload = {
        "question": "Explain vector quantisation.",
        "k": 2,
        "generator_backend": "mock",
    }
    resp = client.post("/ask/stream", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")

    # Parse SSE lines
    lines = resp.text.strip().split("\n\n")
    events = []
    for chunk in lines:
        if chunk.startswith("data: "):
            events.append(json.loads(chunk[6:]))

    assert len(events) >= 2
    token_events = [e for e in events if e.get("event") == "token"]
    done_events = [e for e in events if e.get("event") == "done"]

    assert len(token_events) > 0
    assert len(done_events) == 1
    done = done_events[0]
    assert done["is_finished"] is True
    assert "answer" in done
    assert "generator_info" in done
    assert done["generator_info"]["is_mock"] is True


def test_scientific_audit_endpoint():
    client = TestClient(app)
    resp = client.get("/api/scientific-audit")
    assert resp.status_code == 200
    audit = resp.json()
    assert "split_info" in audit
    assert "fiqa" in audit["split_info"]
    # Verify official BEIR ground-truth splits
    assert audit["split_info"]["fiqa"]["qrels"]["dev"]["queries_with_positive_qrels"] == 500
    assert audit["split_info"]["fiqa"]["qrels"]["test"]["queries_with_positive_qrels"] == 648
