.PHONY: help up down data qa index benchmark loadtest plots test lint typecheck smoke reproduce clean-results

PYTHON ?= python

help:
	@echo "RAGScaleBench Build and Execution Commands:"
	@echo "  make up            - Launch Qdrant and API services via Docker Compose"
	@echo "  make down          - Stop all running benchmark containers"
	@echo "  make data          - Acquire legal corpus documents and generate manifest lock"
	@echo "  make qa            - Generate synthetic dev QA and validate locked test sets"
	@echo "  make index         - Build baseline vector index in Qdrant"
	@echo "  make benchmark     - Execute the full staged research experiment grid"
	@echo "  make loadtest      - Execute k6 arrival-rate saturation benchmarks"
	@echo "  make plots         - Generate all 10 publication-quality research figures"
	@echo "  make test          - Run unit and integration test suites"
	@echo "  make lint          - Run Ruff linter and formatter checks"
	@echo "  make typecheck     - Run Mypy static type verification"
	@echo "  make smoke         - Fast end-to-end smoke test on 200 documents and mock LLM"
	@echo "  make reproduce     - Single-command clean-room reproduction pipeline"
	@echo "  make clean-results - Reset output CSV and plots for a fresh run"

up:
	docker compose up -d qdrant api

down:
	docker compose down

data:
	$(PYTHON) -m ragbench.data.prepare --config configs/default.yaml

qa:
	$(PYTHON) -m ragbench.qa.generate --count 100 --documents data/processed/documents.parquet --output data/qa/dev.generated.jsonl
	$(PYTHON) -m ragbench.qa.validate data/qa/dev.generated.jsonl --documents data/processed/documents.parquet --output data/qa/dev.jsonl
	$(PYTHON) -m ragbench.qa.validate data/qa/test.jsonl --documents data/processed/documents.parquet

index:
	$(PYTHON) -m ragbench.experiments.runner --config configs/default.yaml --stage representation --max-docs 1000

benchmark:
	$(PYTHON) -m ragbench.experiments.runner --grid configs/experiments.yaml --stage all

loadtest:
	@if command -v k6 >/dev/null 2>&1; then \
		k6 run load/retrieve.js; \
		k6 run load/ask.js; \
	else \
		echo "k6 not found. Please install k6 (https://k6.io) to run load saturation tests."; \
	fi

plots:
	$(PYTHON) scripts/plot_results.py --summary results/summary.csv --output plots

test:
	pytest -v tests/

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy ragbench app

smoke:
	@echo "Executing fast architecture smoke test..."
	$(PYTHON) -m ragbench.data.prepare --config configs/smoke.yaml --max-records 200
	$(PYTHON) -m ragbench.experiments.runner --smoke --max-docs 200
	$(PYTHON) scripts/plot_results.py --summary results/summary.csv --output plots
	$(PYTHON) scripts/generate_report.py --summary results/summary.csv --output results/REPORT.md
	@echo "Smoke test passed successfully!"

reproduce:
	bash scripts/reproduce.sh

clean-results:
	rm -rf results/runs/ results/summary.csv plots/*.png results/REPORT.md
	@echo "Results cleaned."