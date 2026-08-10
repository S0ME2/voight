.PHONY: help sync start start-dev zip mibombo clean_start local-validation docker-up docker-up-d docker-down docker-down-v docker-logs docker-logs-clean docker-logs-copy docker-shell docker-ps docker-restart docker-cpu-build docker-cpu-run docker-cpu-up docker-cpu-up-d docker-cpu-test docker-gpu-build docker-gpu-test

.SILENT:

help:
	echo "sync:      Sync locked dependencies with uv"
	echo "start:     Start the FastAPI server"
	echo "start-dev: Start the FastAPI server with reload"
	echo "docker-up:        Build and run the Compose profile selected by .env"
	echo "docker-up-d:      Build and run it in the background"
	echo "docker-down:      Stop and remove Compose containers and network"
	echo "docker-down-v:    docker-down plus the persistent artifact volume"
	echo "docker-logs:      Follow service logs; docker-logs-clean deletes artifacts"
	echo "docker-logs-copy: Copy /app/logs from the CPU container to this PC"
	echo "docker-shell:     Open a shell in the running CPU app container"
	echo "docker-restart:   Restart the selected Compose profile"
	echo "docker-cpu-build: Build the CPU image, including pinned model assets"
	echo "docker-cpu-run:   Alias for docker-up; docker-cpu-up[-d] forces CPU"
	echo "docker-cpu-test:  Run the CPU test suite in the CPU image"
	echo "local-validation: Write CPU-only extraction and batch baseline JSON"
	echo "docker-gpu-build: Build the GPU image on the V100 server only"
	echo "docker-gpu-test:  Run GPU smoke checks on the V100 server only"

sync:
	uv sync --frozen

start:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8888

start-dev:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8888 --reload

zip:
	zip -r code.zip ./app -x "**cache**" "**venv**" "**vscode**" ".**" "**.md**" "dataset/**"

mibombo:
	echo "No runtime text log is used; OCR artifacts are controlled by LOGGING and LOG_DIR."

clean_start: mibombo start

local-validation:
	uv run python -m scripts.local_validation

docker-cpu-build:
	docker build --target cpu -t voight:cpu .

docker-up:
	docker compose --env-file .env up --build

docker-up-d:
	docker compose --env-file .env up --build -d

docker-down:
	docker compose --env-file .env down

docker-down-v:
	docker compose --env-file .env down --volumes

docker-logs:
	docker compose --env-file .env logs --follow

docker-logs-clean:
	docker compose --env-file .env exec -T cpu sh -lc 'find /app/logs -mindepth 1 -delete'

docker-logs-copy:
	mkdir -p logs-from-container
	docker compose --env-file .env cp cpu:/app/logs ./logs-from-container

docker-shell:
	docker compose --env-file .env exec cpu sh

docker-ps:
	docker compose --env-file .env ps

docker-restart:
	docker compose --env-file .env restart

docker-cpu-run: docker-up

docker-cpu-up:
	docker compose --env-file .env --profile cpu up --build

docker-cpu-up-d:
	docker compose --env-file .env --profile cpu up --build -d

docker-cpu-test:
	docker build --target cpu-test -t voight:cpu-test .
	docker run --rm voight:cpu-test

docker-gpu-build:
	docker build --target gpu -t voight:gpu .

docker-gpu-test:
	docker compose --env-file .env --profile gpu run --rm gpu python -c 'from app.config import Settings; from app.models import Models; settings = Settings.from_env(); assert settings.runtime.target == "gpu"; print(Models(settings).readiness())'
