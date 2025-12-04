# Makefile for SmartPantry

.PHONY: help setup install test lint format clean run docker-build docker-run deploy-hf

help:
	@echo "SmartPantry - Available Commands:"
	@echo ""
	@echo "  make setup          - Initial project setup (installs uv)"
	@echo "  make install        - Install dependencies with uv"
	@echo "  make install-dev    - Install with dev dependencies"
	@echo "  make install-all    - Install all optional dependencies"
	@echo "  make sync           - Sync dependencies with uv"
	@echo "  make lock           - Generate requirements.txt from pyproject.toml"
	@echo "  make test           - Run tests"
	@echo "  make test-cov       - Run tests with coverage"
	@echo "  make lint           - Run linting"
	@echo "  make format         - Format code"
	@echo "  make clean          - Clean temporary files"
	@echo "  make run            - Run Gradio app"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run Docker container"
	@echo "  make deploy-hf      - Deploy to Hugging Face Spaces"
	@echo ""

setup:
	@echo "🥗 Setting up SmartPantry..."
	chmod +x setup.sh
	./setup.sh

install:
	@echo "📦 Installing dependencies with uv..."
	uv pip install -e .

install-dev:
	@echo "📦 Installing development dependencies with uv..."
	uv pip install -e ".[dev]"

install-all:
	@echo "📦 Installing all dependencies with uv..."
	uv pip install -e ".[all]"

sync:
	@echo "🔄 Syncing dependencies with uv..."
	uv pip sync

lock:
	@echo "🔒 Generating uv.lock..."
	uv pip compile pyproject.toml -o requirements.txt
	@echo "✅ Generated requirements.txt from pyproject.toml"

test:
	@echo "🧪 Running tests..."
	pytest -v

test-cov:
	@echo "🧪 Running tests with coverage..."
	pytest --cov=src --cov-report=html --cov-report=term-missing

lint:
	@echo "🔍 Running linting..."
	flake8 src/ app/ tests/ --max-line-length=100 --exclude=venv,__pycache__

format:
	@echo "✨ Formatting code..."
	black src/ app/ tests/ --line-length=100
	isort src/ app/ tests/

type-check:
	@echo "🔎 Running type checker..."
	mypy src/ --ignore-missing-imports

clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage

run:
	@echo "🚀 Starting FastAPI backend..."
	uvicorn app.api_extended:app --reload --port 8001

test-api:
	@echo "🧪 Testing API..."
	python test_api.py

run-dev:
	@echo "🚀 Starting FastAPI backend in development mode..."
	uvicorn app.api_extended:app --reload --port 8001 --log-level debug

docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t smartpantry:latest .

docker-run:
	@echo "🐳 Running Docker container..."
	docker run -p 8001:8001 -v $(PWD)/data:/app/data -v $(PWD)/models:/app/models smartpantry:latest

docker-compose-up:
	@echo "🐳 Starting with Docker Compose..."
	docker-compose up -d

docker-compose-down:
	@echo "🐳 Stopping Docker Compose..."
	docker-compose down

docker-compose-logs:
	@echo "🐳 Viewing Docker Compose logs..."
	docker-compose logs -f

deploy-hf:
	@echo "🤗 Preparing Hugging Face Spaces deployment..."
	@echo "✅ Ready to deploy! Follow these steps:"
	@echo ""
	@echo "1. Create a Space on Hugging Face:"
	@echo "   https://huggingface.co/spaces"
	@echo "   - SDK: Docker"
	@echo "   - Hardware: CPU basic (free) or GPU (paid)"
	@echo ""
	@echo "2. Add HF remote and push:"
	@echo "   huggingface-cli login"
	@echo "   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/smartpantry"
	@echo "   git push hf main"
	@echo ""
	@echo "3. Wait for build (5-10 minutes)"
	@echo "4. Access your Space at:"
	@echo "   https://huggingface.co/spaces/YOUR_USERNAME/smartpantry"

download-models:
	@echo "📥 Downloading YOLOv8 models..."
	mkdir -p models/yolo
	wget -q --show-progress https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolo/yolov8n.pt
	@echo "✅ YOLOv8n downloaded!"

notebooks:
	@echo "📓 Starting Jupyter Lab..."
	jupyter lab

check: lint type-check test
	@echo "✅ All checks passed!"

ci: install-dev check
	@echo "✅ CI pipeline completed!"

.PHONY: install install-dev install-all sync lock run-dev docker-compose-up docker-compose-down docker-compose-logs download-models notebooks check ci

