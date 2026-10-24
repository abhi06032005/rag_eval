# RAGScaleBench — Docker Deployment & Containerization Guide

This document describes how to deploy the complete RAGScaleBench stack (Qdrant Vector DB, FastAPI Serving Layer, Next.js Web Platform, and optional LLM engines) using Docker and Docker Compose.

---

## 1. Architecture & Service Topology

The Docker Compose configuration ([`compose.yaml`](compose.yaml)) defines five coordinated services:

```
                  ┌───────────────────────────────┐
                  │       Browser Client          │
                  │   (http://localhost:3000)     │
                  └──────────────┬────────────────┘
                                 │
                 HTTP / SSE      ▼
                  ┌───────────────────────────────┐
                  │   ragscalebench-web           │
                  │   (Next.js 16 on port 3000)   │
                  └──────────────┬────────────────┘
                                 │
                 REST API        ▼
                  ┌───────────────────────────────┐
                  │   ragscalebench-api           │
                  │   (FastAPI on port 8000)      │
                  └───────┬───────────────┬───────┘
                          │               │
        gRPC / HTTP (6333)│               │ HTTP (11434 / 8000)
                          ▼               ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│   ragscalebench-qdrant        │   │ ragscalebench-ollama / vllm   │
│   (Qdrant Rust Vector DB)     │   │ (Local / GPU Generator)       │
└───────────────────────────────┘   └───────────────────────────────┘
```

---

## 2. Quickstart Deployment

### Standard Stack (Qdrant + FastAPI API + Next.js Web App)

```bash
# Build and launch services in background
docker compose up -d --build qdrant api web

# Verify container health
docker compose ps
```

Services will be accessible at:
- **Next.js Web Application:** [`http://localhost:3000`](http://localhost:3000)
- **FastAPI OpenAPI Swagger:** [`http://localhost:8000/docs`](http://localhost:8000/docs)
- **Qdrant Vector Database:** [`http://localhost:6333/dashboard`](http://localhost:6333/dashboard)
- **Prometheus Metrics:** [`http://localhost:8000/metrics`](http://localhost:8000/metrics)

---

## 3. Optional Local LLM Deployment (Ollama Profile)

To include a local Ollama container running on CPU/GPU:

```bash
# Launch with Ollama service profile
docker compose --profile ollama up -d --build

# Pull the default Llama 3.2 model into the container
docker exec -it ragscalebench-ollama ollama pull llama3.2:3b
```

Once pulled, select **"Local Ollama"** in the Next.js QA Playground or pass `"generator_backend": "ollama"` to the `/ask` endpoint.

---

## 4. Optional GPU Serving Deployment (vLLM Profile)

To deploy with high-throughput NVIDIA GPU generation via vLLM:

```bash
# Set your Hugging Face token in environment
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"

# Launch with GPU vLLM profile
docker compose --profile vllm up -d --build
```

---

## 5. Health Probes & Telemetry

Each container defines automated healthchecks:
- `qdrant`: TCP port 6333 socket probe every 5s.
- `api`: HTTP `GET http://localhost:8000/healthz` every 10s.
- `web`: Depends on `api` reaching `service_healthy`.

To view live container logs:
```bash
docker compose logs -f api
docker compose logs -f web
```
