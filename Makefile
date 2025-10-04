.PHONY: setup run ingest test eval format lint docker-build clean

setup:
	@echo "Setting up environment..."
	python3.11 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"
	mkdir -p data/kb data/index models
	@echo "✓ Setup complete. Activate: source .venv/bin/activate"

run:
	@echo "Starting FastAPI server..."
	.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

ingest:
	@echo "Ingesting documents..."
	.venv/bin/python -m app.rag_pipeline --ingest

chat:
	@echo "Example chat request..."
	curl -X POST http://localhost:8000/chat \
		-H "Content-Type: application/json" \
		-d '{"query": "What is this project about?"}'

test:
	@echo "Running tests with coverage..."
	.venv/bin/pytest -v

eval:
	@echo "Running evaluation harness..."
	.venv/bin/python eval/run_eval.py

format:
	.venv/bin/black app/ tests/ eval/
	.venv/bin/ruff --fix app/ tests/ eval/

lint:
	.venv/bin/ruff check app/ tests/ eval/
	.venv/bin/black --check app/ tests/ eval/

docker-build:
	docker build -f docker/Dockerfile -t llm-rag-chatbot:latest .

clean:
	rm -rf .venv __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete