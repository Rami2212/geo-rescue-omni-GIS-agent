# ─── GeoRescue — Omni GIS Agent ─────────────────────────────────────────────
# Convenience targets for development and deployment.
# Usage: make <target>

.PHONY: help install install-ml env api app all train enrich lint clean

PYTHON   := python
PIP      := $(PYTHON) -m pip
UVICORN  := uvicorn
STREAMLIT := streamlit

help:
	@echo ""
	@echo "  GeoRescue — Omni GIS Agent"
	@echo ""
	@echo "  make install     Install UI + agent dependencies"
	@echo "  make install-ml  Install ML serving dependencies (GPU required)"
	@echo "  make env         Create .env from .env.example (if .env missing)"
	@echo "  make api         Start the GIS + Vision API server (port 9000)"
	@echo "  make app         Start the Streamlit UI (port 8501)"
	@echo "  make all         Start both API and app in parallel"
	@echo "  make lint        Run ruff linter on all Python files"
	@echo "  make enrich      Generate training labels from GIS sample data"
	@echo "  make train       Fine-tune Qwen2-VL-7B with LoRA on collected data"
	@echo "  make clean       Remove __pycache__ and .pyc files"
	@echo ""

# ── Setup ────────────────────────────────────────────────────────────────────

install:
	$(PIP) install -r requirements.txt

install-ml:
	cd ml_serving && $(PIP) install -r requirements.txt

env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo ".env created from .env.example — fill in your values."; \
	else \
		echo ".env already exists — skipping."; \
	fi

# ── Services ─────────────────────────────────────────────────────────────────

api:
	@echo "Starting GIS + Vision API on port 9000..."
	cd ml_serving && $(UVICORN) api.app:app --host 0.0.0.0 --port 9000 --reload

app:
	@echo "Starting Streamlit UI on port 8501..."
	$(STREAMLIT) run app.py

all:
	@echo "Starting both services (Ctrl+C to stop)..."
	@$(MAKE) api & $(MAKE) app; wait

# ── Training ─────────────────────────────────────────────────────────────────

DATASET_INDEX := data/dataset_index.json
TRAIN_OUTPUT  := final3

enrich:
	@echo "Enriching training labels from GIS samples..."
	cd ml_serving && $(PYTHON) training/run_training.py enrich --dataset-index $(DATASET_INDEX)

train:
	@echo "Fine-tuning Qwen2-VL-7B with LoRA (AMD MI300X / ROCm required)..."
	cd ml_serving && $(PYTHON) training/run_training.py train \
		--dataset-index $(DATASET_INDEX) \
		--output-dir $(TRAIN_OUTPUT) \
		--epochs 3 \
		--batch-size 2 \
		--enrich-labels

# ── Code quality ─────────────────────────────────────────────────────────────

lint:
	$(PYTHON) -m ruff check agents/ georescue/ app.py

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	find . -name "*.pyo" -delete 2>/dev/null; true
