# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/home/voight PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True MODEL_DIR=/opt/voight/models
WORKDIR /app
RUN groupadd --system voight && useradd --system --gid voight --home-dir "$HOME" --create-home voight && mkdir -p /app/logs "$MODEL_DIR" && chown -R voight:voight /app/logs "$HOME" "$MODEL_DIR" && apt-get update && apt-get install --no-install-recommends -y git libgl1 libglib2.0-0 libgomp1 libturbojpeg0 && rm -rf /var/lib/apt/lists/*
COPY requirements/ /app/requirements/

FROM base AS cpu-deps
RUN pip install --no-cache-dir --no-deps -r requirements/cpu.lock

FROM base AS gpu-deps
RUN pip install --no-cache-dir -r requirements/gpu.txt

FROM cpu-deps AS cpu-assets
COPY app/ /app/app/
COPY config/ /app/config/
COPY scripts/models/prepare.py /app/scripts/models/prepare.py
ENV PYTHONPATH=/app RUNTIME_TARGET=cpu PRELOAD=false
RUN python scripts/models/prepare.py && chown -R voight:voight "$MODEL_DIR"

FROM gpu-deps AS gpu-assets
COPY app/ /app/app/
COPY config/ /app/config/
COPY scripts/models/prepare.py /app/scripts/models/prepare.py
# Cache assets without requiring a GPU during docker build; runtime selection is
# applied only in the final image.
ENV PYTHONPATH=/app RUNTIME_TARGET=cpu PRELOAD=false
RUN python scripts/models/prepare.py && chown -R voight:voight "$MODEL_DIR"

FROM cpu-assets AS cpu
COPY --chown=voight:voight app/ /app/app/
COPY --chown=voight:voight config/ /app/config/
USER voight
ENV RUNTIME_TARGET=cpu PRELOAD=false
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/v1/health/ready').read()"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

FROM cpu AS cpu-test
USER root
COPY scripts/ /app/scripts/
COPY tests/ /app/tests/
COPY README.md .env.example /app/
COPY docs/ /app/docs/
COPY .git/ /app/.git/
COPY annotation_input/ /app/annotation_input/
COPY annotations/ /app/annotations/
COPY benchmarks/ /app/benchmarks/
COPY archive/ /app/archive/
CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]

FROM gpu-assets AS gpu
COPY --chown=voight:voight app/ /app/app/
COPY --chown=voight:voight config/ /app/config/
USER voight
ENV RUNTIME_TARGET=gpu GPU_ID=0 PRELOAD=false
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/v1/health/ready').read()"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
