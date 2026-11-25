.PHONY: help install dev test lint format typecheck clean build up down logs shell migrate generate-data train evaluate demo

# Default target
help:
	@echo "Document Intelligence - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install      Install dependencies"
	@echo "  dev          Start development environment"
	@echo ""
	@echo "Development:"
	@echo "  test         Run tests"
	@echo "  lint         Run ruff linting"
	@echo "  format       Format code with ruff"
	@echo "  typecheck    Run mypy type checking"
	@echo "  shell        Open Python shell with app context"
	@echo ""
	@echo "Data & Training:"
	@echo "  generate-data  Generate synthetic invoice dataset"
	@echo "  train          Train LayoutLMv3 model"
	@echo "  evaluate       Run evaluation suite"
	@echo ""
	@echo "Services:"
	@echo "  up             Start all services (docker-compose)"
	@echo "  down           Stop all services"
	@echo "  logs           View service logs"
	@echo "  migrate        Run database migrations"
	@echo ""
	@echo "Production:"
	@echo "  build          Build Docker images"
	@echo "  demo           Run Streamlit demo"
	@echo "  clean          Clean build artifacts"

# Install dependencies
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .[dev]
	pre-commit install

# Development environment
dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
	@echo "Services started:"
	@echo "  API: http://localhost:8000/docs"
	@echo "  MinIO: http://localhost:9001"
	@echo "  MLflow: http://localhost:5000"
	@echo "  Jupyter: http://localhost:8888"

# Testing
test:
	pytest tests/ -v --tb=short --cov=document_intelligence --cov-report=term-missing

test-unit:
	pytest tests/test_baseline.py -v --tb=short

test-integration:
	pytest tests/test_api.py -v --tb=short

test-load:
	locust -f tests/load/locustfile.py --host=http://localhost:8000

# Code quality
lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/

# Pre-commit
pre-commit:
	pre-commit run --all-files

# Shell
shell:
	ipython --TerminalInteractiveShell.colors="Linux"

# Database
migrate:
	alembic upgrade head

migrate-create:
	alembic revision --autogenerate -m "$(msg)"

# Docker
up:
	docker-compose up -d

down:
	docker-compose down

down-volumes:
	docker-compose down -v

logs:
	docker-compose logs -f

logs-api:
	docker-compose logs -f api

logs-worker:
	docker-compose logs -f worker

build:
	docker-compose build

build-api:
	docker-compose build api

build-worker:
	docker-compose build worker

# Data generation
generate-data:
	python -m document_intelligence.data.synthetic.generate --output data/synthetic --count 1000

generate-data-small:
	python -m document_intelligence.data.synthetic.generate --output data/synthetic --count 100 --no-noise

# Training
train:
	python -m document_intelligence.training.train \
		--train-data data/synthetic/train_annotations.json \
		--val-data data/synthetic/val_annotations.json \
		--output-dir models/layoutlmv3-finetuned \
		--epochs 10 \
		--batch-size 4 \
		--lr 3e-5

train-quick:
	python -m document_intelligence.training.train \
		--train-data data/synthetic/train_annotations.json \
		--val-data data/synthetic/val_annotations.json \
		--output-dir models/layoutlmv3-finetuned \
		--epochs 3 \
		--batch-size 2 \
		--lr 5e-5

# Export model to ONNX
export-onnx:
	python -c "
from document_intelligence.training.train import export_to_onnx
export_to_onnx('models/layoutlmv3-finetuned', 'models/layoutlmv3-finetuned/model.onnx')
"

# Evaluation
evaluate:
	python -m document_intelligence.evaluation.metrics \
		--predictions data/test/baseline_predictions.json \
		--ground-truth data/test/ground_truth.json \
		--output evaluation/baseline_report.html \
		--model-name "baseline"

evaluate-model:
	python -m document_intelligence.evaluation.metrics \
		--predictions data/test/model_predictions.json \
		--ground-truth data/test/ground_truth.json \
		--output evaluation/model_report.html \
		--model-name "layoutlmv3" \
		--baseline evaluation/baseline_results.json

# Demo
demo:
	streamlit run demo/app.py --server.port 8501 --server.address 0.0.0.0

# Clean
clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov coverage.xml
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# CI simulation
ci: lint typecheck test
	@echo "All CI checks passed!"

# Full setup for new developers
setup: install dev generate-data
	@echo "Setup complete! Run 'make train' to train the model."