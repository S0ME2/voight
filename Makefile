.PHONY: help sync start start-dev zip mibombo clean_start

.SILENT:

help:
	echo "sync:      Sync locked dependencies with uv"
	echo "start:     Start the FastAPI server"
	echo "start-dev: Start the FastAPI server with reload"

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
