# Voight Makefile.
#
# Local commands are CPU-only. Docker uses the CPU service by default; GPU
# targets are explicit and intended only for the V100 deployment host.

ENV_FILE ?= .env
UV ?= uv
UV_RUN = $(UV) run --no-sync
IMAGE ?= voight
CPU_IMAGE ?= $(IMAGE):cpu
CPU_TEST_IMAGE ?= $(IMAGE):cpu-test
GPU_IMAGE ?= $(IMAGE):gpu
LOCAL_HOST ?= 0.0.0.0
LOCAL_PORT ?= 8888

# These variables keep custom image tags and the service env_file in sync with
# the Make targets. Direct `docker compose` use defaults to the checked-in
# example values.
COMPOSE = CPU_IMAGE=$(CPU_IMAGE) GPU_IMAGE=$(GPU_IMAGE) VOIGHT_ENV_FILE=$(ENV_FILE) docker compose --env-file $(ENV_FILE)
COMPOSE_GPU = $(COMPOSE) --profile gpu

.PHONY: help
help: ## Show all commands grouped by section
	@awk 'BEGIN {FS = ":.*## |##@ "; print "Voight commands:\n"} /^##@/ {printf "\n%s\n", $$2} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-30s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

##@ Setup
.PHONY: install
install: ## Install the locked local CPU development environment
	$(UV) venv --allow-existing
	$(UV) pip sync --reinstall-package onnxruntime requirements/cpu.lock

##@ Local API
.PHONY: run run-dev
run: ## Start the local CPU API (override LOCAL_PORT to change the port)
	RUNTIME_TARGET=cpu OCR_DEVICE=cpu MODEL_DIR= $(UV_RUN) uvicorn app.main:app --host $(LOCAL_HOST) --port $(LOCAL_PORT)

run-dev: ## Start the local CPU API with auto-reload
	RUNTIME_TARGET=cpu OCR_DEVICE=cpu MODEL_DIR= $(UV_RUN) uvicorn app.main:app --host $(LOCAL_HOST) --port $(LOCAL_PORT) --reload

##@ Quality gates
.PHONY: test coverage check validate
test: ## Run the complete CPU-safe test suite
	RUNTIME_TARGET=cpu OCR_DEVICE=cpu MODEL_DIR= $(UV_RUN) python -m unittest discover -s tests -v

coverage: ## Run the complete suite with the app coverage floor
	RUNTIME_TARGET=cpu OCR_DEVICE=cpu MODEL_DIR= $(UV_RUN) coverage erase
	RUNTIME_TARGET=cpu OCR_DEVICE=cpu MODEL_DIR= $(UV_RUN) coverage run --branch --source=app -m unittest discover -s tests -v
	RUNTIME_TARGET=cpu OCR_DEVICE=cpu MODEL_DIR= $(UV_RUN) coverage report --fail-under=$${COVERAGE_MIN:-69}

check: ## Compile Python, check the lock, and reject whitespace errors
	PYTHONDONTWRITEBYTECODE=1 $(UV_RUN) python -m compileall -q app scripts tests benchmarks archive
	$(UV) lock --check
	git diff --check

validate: ## Write the CPU extraction and true-batch baseline
	RUNTIME_TARGET=cpu OCR_DEVICE=cpu MODEL_DIR= $(UV_RUN) python -m scripts.validation.local

##@ Docker (CPU)
.PHONY: docker-cpu-build docker-cpu-up docker-cpu-up-d docker-cpu-down docker-cpu-logs \
        docker-cpu-shell docker-cpu-test docker-cpu-artifacts-copy docker-cpu-artifacts-clean
docker-cpu-build: ## Build the CPU image with pinned models inside it
	docker build --target cpu -t $(CPU_IMAGE) .

docker-cpu-up: env-check ## Run the built CPU service in the foreground
	$(COMPOSE) up cpu

docker-cpu-up-d: env-check ## Run the built CPU service in the background
	$(COMPOSE) up -d cpu

docker-cpu-down: env-check ## Stop Compose and remove its volumes/networks
	$(COMPOSE) down --volumes --remove-orphans

docker-cpu-logs: env-check ## Follow CPU service logs
	$(COMPOSE) logs --follow cpu

docker-cpu-shell: env-check ## Open a shell in the running CPU container
	$(COMPOSE) exec cpu sh

docker-cpu-test: ## Run CPU contract, batching, API, and model-cache tests in Docker
	docker build --target cpu-test -t $(CPU_TEST_IMAGE) .
	docker run --rm $(CPU_TEST_IMAGE)

docker-cpu-artifacts-copy: env-check ## Copy persisted container artifacts into outputs/
	mkdir -p outputs/docker-artifacts
	$(COMPOSE) cp cpu:/app/logs/. outputs/docker-artifacts/

docker-cpu-artifacts-clean: env-check ## Delete persisted CPU artifacts after copying anything needed
	$(COMPOSE) exec -T cpu sh -c 'find /app/logs -mindepth 1 -delete'

##@ Docker (GPU)
# The GPU profile is meant for the V100 server only.
.PHONY: docker-gpu-build docker-gpu-up docker-gpu-up-d docker-gpu-down docker-gpu-logs docker-gpu-test
docker-gpu-build: ## Build the GPU image on the V100 server only
	docker build --target gpu -t $(GPU_IMAGE) .

docker-gpu-up: env-check ## Run the GPU service on the V100 server only
	$(COMPOSE_GPU) up gpu

docker-gpu-up-d: env-check ## Run the GPU service detached on the V100 server only
	$(COMPOSE_GPU) up -d gpu

docker-gpu-down: env-check ## Stop GPU Compose and remove its volumes/networks
	$(COMPOSE_GPU) down --volumes --remove-orphans

docker-gpu-logs: env-check ## Follow GPU service logs
	$(COMPOSE_GPU) logs --follow gpu

docker-gpu-test: env-check ## Run GPU readiness on the V100 server only
	$(COMPOSE_GPU) run --rm gpu python -c 'from app.config import Settings; from app.models import Models; print(Models(Settings.from_env()).readiness())'

##@ Models
.PHONY: models-info models-rebuild-cpu
models-info: ## Print the model manifest baked into the CPU image
	docker run --rm --entrypoint cat $(CPU_IMAGE) /opt/voight/models/voight-models.json

models-rebuild-cpu: ## Force a fresh CPU dependency and model download
	docker build --no-cache --target cpu -t $(CPU_IMAGE) .

##@ Dataset
.PHONY: dataset-annotate dataset-summary dataset-validate dataset-export
dataset-annotate: ## Start or resume local ground-truth annotation
	$(UV_RUN) python scripts/dataset/annotate.py

dataset-summary: ## Show local dataset annotation progress
	$(UV_RUN) python scripts/dataset/annotate.py --summary

dataset-validate: ## Validate local dataset structure and annotations
	$(UV_RUN) python scripts/dataset/annotate.py --validate

dataset-export: ## Export local annotations to ignored JSONL output
	$(UV_RUN) python scripts/dataset/annotate.py --export

##@ Benchmarks
.PHONY: benchmark-cpu benchmark-recognition benchmark-verification benchmark-verification-batch-sizes benchmark-verification-followup benchmark-verification-compare benchmark-verification-report
benchmark-cpu: ## Benchmark a running CPU API with committed fixtures
	$(UV_RUN) python benchmarks/maintained/benchmark_batch_complexity.py --runtime cpu --repeats 3

benchmark-recognition: ## Benchmark cached recognizer batch sizes locally
	@test -n "$$MODEL_DIR" || { echo "MODEL_DIR must point to a prepared local model cache"; exit 2; }
	$(UV_RUN) python benchmarks/maintained/text_recognition_batch_benchmark.py

benchmark-verification: ## Run the fresh-process whole-document verification baseline
	@test -n "$$MODEL_DIR" || { echo "MODEL_DIR must point to a prepared CPU model cache"; exit 2; }
	$(UV_RUN) python benchmarks/maintained/verification_benchmark.py --model-dir "$$MODEL_DIR"

benchmark-verification-batch-sizes: ## Sweep internal verification tensor batch limits on CPU
	@test -n "$$MODEL_DIR" || { echo "MODEL_DIR must point to a prepared CPU model cache"; exit 2; }
	$(UV_RUN) python benchmarks/maintained/verification_batch_size_benchmark.py --model-dir "$$MODEL_DIR"

benchmark-verification-followup: ## Diagnose aggregate verification complexity and repeated-request memory
	@test -n "$$MODEL_DIR" || { echo "MODEL_DIR must point to a prepared CPU model cache"; exit 2; }
	$(UV_RUN) python -m benchmarks.verification.diagnostic --model-dir "$$MODEL_DIR"

benchmark-verification-compare: ## Compare two verification benchmark result directories
	@test -n "$$OLD_RUN" -a -n "$$NEW_RUN" || { echo "OLD_RUN and NEW_RUN are required"; exit 2; }
	$(UV_RUN) python -m benchmarks.verification.compare "$$OLD_RUN" "$$NEW_RUN"

benchmark-verification-report: ## Regenerate a verification summary from raw artifacts
	@test -n "$$RUN_DIR" || { echo "RUN_DIR is required"; exit 2; }
	$(UV_RUN) python -m benchmarks.verification.report "$$RUN_DIR"

##@ Maintenance
.PHONY: env-check clean
env-check: ## Fail fast when the Compose env file is missing
	@test -f "$(ENV_FILE)" || { echo "Missing $(ENV_FILE). Run: cp .env.example .env"; exit 2; }

clean: ## Remove Python caches only; datasets, models, logs, and outputs are preserved
	find app scripts tests benchmarks archive -type d -name __pycache__ -prune -exec rm -r {} +
