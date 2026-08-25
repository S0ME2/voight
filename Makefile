# Voight Makefile.
#
# Conventions:
#   - Local Python entry points run through `uv run --no-sync` ($(UV)) so every
#     command uses the locked environment without triggering an implicit sync.
#   - Quality gates (`test`, `check`, `validate`) clear MODEL_DIR so they never
#     depend on a prepared model cache; benchmarks that need real weights
#     require MODEL_DIR explicitly.
#   - Docker targets split into CPU (the default workhorse) and GPU (V100
#     server only); both read the same $(ENV_FILE).

ENV_FILE ?= .env
UV = uv run --no-sync
COMPOSE = docker compose --env-file $(ENV_FILE)
COMPOSE_GPU = $(COMPOSE) --profile gpu

.PHONY: help
help: ## Show all commands grouped by section
	@awk 'BEGIN {FS = ":.*## |##@ "; print "Voight commands:\n"} /^##@/ {printf "\n%s\n", $$2} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-30s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

##@ Setup
.PHONY: install
install: ## Install the locked local CPU development environment
	uv venv --allow-existing
	uv pip sync --reinstall-package onnxruntime requirements/cpu.lock

##@ Local API
.PHONY: run run-dev
run: ## Start the local CPU API on port 8888
	MODEL_DIR= $(UV) uvicorn app.main:app --host 0.0.0.0 --port 8888

run-dev: ## Start the local CPU API with auto-reload
	MODEL_DIR= $(UV) uvicorn app.main:app --host 0.0.0.0 --port 8888 --reload

##@ Quality gates
.PHONY: test coverage check validate
test: ## Run the complete CPU-safe test suite
	MODEL_DIR= $(UV) python -m unittest discover -s tests -v

coverage: ## Run the complete suite with the committed coverage floor
	MODEL_DIR= $(UV) coverage erase
	MODEL_DIR= $(UV) coverage run --branch -m unittest discover -s tests -v
	MODEL_DIR= $(UV) coverage report --fail-under=$${COVERAGE_MIN:-69}

check: ## Compile Python, check the lock, and reject whitespace errors
	PYTHONDONTWRITEBYTECODE=1 $(UV) python -m compileall -q app scripts tests benchmarks archive
	uv lock --check
	git diff --check

validate: ## Write the CPU extraction and true-batch baseline
	MODEL_DIR= $(UV) python -m scripts.validation.local

##@ Docker (CPU)
.PHONY: docker-cpu-build docker-cpu-up docker-cpu-up-d docker-cpu-down docker-cpu-logs \
        docker-cpu-shell docker-cpu-test docker-cpu-artifacts-copy docker-cpu-artifacts-clean
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

docker-cpu-test: ## Run CPU contract, batching, API, and model-cache tests in Docker
	docker build --target cpu-test -t voight:cpu-test .
	docker run --rm voight:cpu-test

docker-cpu-artifacts-copy: env-check ## Copy persisted container artifacts into outputs/
	mkdir -p outputs/docker-artifacts
	$(COMPOSE) cp cpu:/app/logs/. outputs/docker-artifacts/

docker-cpu-artifacts-clean: env-check ## Delete persisted CPU artifacts after copying anything needed
	$(COMPOSE) exec -T cpu sh -c 'find /app/logs -mindepth 1 -delete'

##@ Docker (GPU)
# The GPU profile is meant for the V100 server only.
.PHONY: docker-gpu-build docker-gpu-up docker-gpu-up-d docker-gpu-down docker-gpu-logs docker-gpu-test
docker-gpu-build: ## Build the GPU image on the V100 server only
	docker build --target gpu -t voight:gpu .

docker-gpu-up: env-check ## Run the GPU service on the V100 server only
	$(COMPOSE_GPU) up gpu

docker-gpu-up-d: env-check ## Run the GPU service detached on the V100 server only
	$(COMPOSE_GPU) up -d gpu

docker-gpu-down: env-check ## Stop the GPU service on the V100 server only
	$(COMPOSE_GPU) down

docker-gpu-logs: env-check ## Follow GPU service logs
	$(COMPOSE_GPU) logs --follow gpu

docker-gpu-test: env-check ## Run GPU readiness on the V100 server only
	$(COMPOSE_GPU) run --rm gpu python -c 'from app.config import Settings; from app.models import Models; print(Models(Settings.from_env()).readiness())'

##@ Models
.PHONY: models-cpu models-info models-rebuild-cpu
models-cpu: docker-cpu-build ## Prepare CPU models by building the image

models-info: ## Print the model manifest baked into the CPU image
	docker run --rm --entrypoint cat voight:cpu /opt/voight/models/voight-models.json

models-rebuild-cpu: ## Force a fresh CPU dependency and model download
	docker build --no-cache --target cpu -t voight:cpu .

##@ Dataset
.PHONY: dataset-annotate dataset-summary dataset-validate dataset-export
dataset-annotate: ## Start or resume local ground-truth annotation
	$(UV) python scripts/dataset/annotate.py

dataset-summary: ## Show local dataset annotation progress
	$(UV) python scripts/dataset/annotate.py --summary

dataset-validate: ## Validate local dataset structure and annotations
	$(UV) python scripts/dataset/annotate.py --validate

dataset-export: ## Export local annotations to ignored JSONL output
	$(UV) python scripts/dataset/annotate.py --export

##@ Benchmarks
.PHONY: benchmark-cpu benchmark-recognition
benchmark-cpu: ## Benchmark a running CPU API with committed fixtures
	$(UV) python benchmarks/maintained/benchmark_batch_complexity.py --runtime cpu --repeats 3

benchmark-recognition: ## Benchmark cached recognizer batch sizes locally
	@test -n "$$MODEL_DIR" || { echo "MODEL_DIR must point to a prepared local model cache"; exit 2; }
	$(UV) python benchmarks/maintained/text_recognition_batch_benchmark.py

##@ Maintenance
.PHONY: env-check clean
env-check: ## Fail fast when the Compose env file is missing
	@test -f $(ENV_FILE) || { echo "Missing $(ENV_FILE). Run: cp .env.example .env"; exit 2; }

clean: ## Remove Python caches only; datasets, models, logs, and outputs are preserved
	find app scripts tests benchmarks archive -type d -name __pycache__ -prune -exec rm -r {} +
