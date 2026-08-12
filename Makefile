ENV_FILE ?= .env
COMPOSE = docker compose --env-file $(ENV_FILE)

.PHONY: help install run run-dev test check validate env-check \
	docker-cpu-build docker-cpu-up docker-cpu-up-d docker-cpu-down docker-cpu-logs docker-cpu-shell docker-cpu-test docker-cpu-artifacts-copy docker-cpu-artifacts-clean \
	docker-gpu-build docker-gpu-up docker-gpu-up-d docker-gpu-down docker-gpu-logs docker-gpu-test \
	models-cpu models-info models-rebuild-cpu dataset-annotate dataset-summary dataset-validate dataset-export \
	benchmark-cpu benchmark-recognition clean

help: ## Show supported commands
	@awk 'BEGIN {FS = ":.*## "; print "Voight commands:\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-28s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the locked local CPU development environment
	uv venv --allow-existing
	uv pip sync --reinstall-package onnxruntime requirements/cpu.lock

run: ## Start the local CPU API on port 8888
	MODEL_DIR= uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port 8888

run-dev: ## Start the local CPU API with reload
	MODEL_DIR= uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port 8888 --reload

test: ## Run the complete CPU-safe test suite
	MODEL_DIR= uv run --no-sync python -m unittest discover -s tests -v

check: ## Compile Python, check the lock, and reject whitespace errors
	PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python -m compileall -q app scripts tests
	uv lock --check
	git diff --check

validate: ## Write the CPU extraction and true-batch baseline
	MODEL_DIR= uv run --no-sync python -m scripts.validation.local

env-check:
	@test -f $(ENV_FILE) || { echo "Missing $(ENV_FILE). Run: cp .env.example .env"; exit 2; }

docker-cpu-build: ## Build the CPU image with pinned models inside it
	docker build --target cpu -t voight:cpu .

docker-cpu-up: env-check ## Run the built CPU service in the foreground
	$(COMPOSE) up cpu

docker-cpu-up-d: env-check ## Run the built CPU service in the background
	$(COMPOSE) up -d cpu

docker-cpu-down: env-check ## Stop the Compose application
	$(COMPOSE) down

docker-cpu-logs: env-check ## Follow CPU service logs
	$(COMPOSE) logs --follow cpu

docker-cpu-shell: env-check ## Open a shell in the running CPU container
	$(COMPOSE) exec cpu sh

docker-cpu-artifacts-copy: env-check ## Copy persisted container artifacts into outputs/
	mkdir -p outputs/docker-artifacts
	$(COMPOSE) cp cpu:/app/logs/. outputs/docker-artifacts/

docker-cpu-artifacts-clean: env-check ## Delete persisted CPU artifacts after copying anything needed
	$(COMPOSE) exec -T cpu sh -c 'find /app/logs -mindepth 1 -delete'

docker-cpu-test: ## Run CPU contract, batching, API, and model-cache tests in Docker
	docker build --target cpu-test -t voight:cpu-test .
	docker run --rm voight:cpu-test

docker-gpu-build: ## Build the GPU image on the V100 server only
	docker build --target gpu -t voight:gpu .

docker-gpu-up: env-check ## Run the GPU service on the V100 server only
	$(COMPOSE) --profile gpu up gpu

docker-gpu-up-d: env-check ## Run the GPU service detached on the V100 server only
	$(COMPOSE) --profile gpu up -d gpu

docker-gpu-down: env-check ## Stop the Compose application on the GPU server
	$(COMPOSE) --profile gpu down

docker-gpu-logs: env-check ## Follow GPU service logs
	$(COMPOSE) --profile gpu logs --follow gpu

docker-gpu-test: env-check ## Run GPU readiness on the V100 server only
	$(COMPOSE) --profile gpu run --rm gpu python -c 'from app.config import Settings; from app.models import Models; print(Models(Settings.from_env()).readiness())'

models-cpu: docker-cpu-build ## Prepare CPU models by building the image

models-info: ## Print the model manifest baked into the CPU image
	docker run --rm --entrypoint cat voight:cpu /opt/voight/models/voight-models.json

models-rebuild-cpu: ## Force a fresh CPU dependency and model download
	docker build --no-cache --target cpu -t voight:cpu .

dataset-annotate: ## Start or resume local ground-truth annotation
	uv run --no-sync python scripts/dataset/annotate.py

dataset-summary: ## Show local dataset annotation progress
	uv run --no-sync python scripts/dataset/annotate.py --summary

dataset-validate: ## Validate local dataset structure and annotations
	uv run --no-sync python scripts/dataset/annotate.py --validate

dataset-export: ## Export local annotations to ignored JSONL output
	uv run --no-sync python scripts/dataset/annotate.py --export

benchmark-cpu: ## Benchmark a running CPU API with committed fixtures
	uv run --no-sync python complexity/benchmark_batch_complexity.py --runtime cpu --repeats 3

benchmark-recognition: ## Benchmark cached recognizer batch sizes locally
	@test -n "$$MODEL_DIR" || { echo "MODEL_DIR must point to a prepared local model cache"; exit 2; }
	uv run --no-sync python scripts/benchmarking/text_recognition_batch_benchmark.py

clean: ## Remove Python caches only; datasets, models, logs, and outputs are preserved
	find app scripts tests complexity tools -type d -name __pycache__ -prune -exec rm -r {} +
