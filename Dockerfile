# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/home/voight PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
WORKDIR /app
RUN groupadd --system voight && useradd --system --gid voight --home-dir "$HOME" --create-home voight && mkdir -p /app/logs && chown voight:voight /app/logs && apt-get update && apt-get install --no-install-recommends -y libgl1 libglib2.0-0 libgomp1 libturbojpeg0 && rm -rf /var/lib/apt/lists/*
COPY requirements/ /app/requirements/

FROM base AS cpu-deps
RUN pip install --no-cache-dir --no-deps -r requirements/cpu.lock

FROM base AS gpu-deps
RUN pip install --no-cache-dir -r requirements/gpu.txt

FROM cpu-deps AS cpu-assets
COPY app/ /app/app/
COPY config/ /app/config/
COPY scripts/cache_models.py /app/scripts/cache_models.py
ENV PYTHONPATH=/app RUNTIME_TARGET=cpu OCR_DEVICE=cpu PRELOAD=false MODEL_DIR=/home/voight/.paddlex
RUN mkdir -p "$MODEL_DIR" && python scripts/cache_models.py

FROM gpu-deps AS gpu-assets
COPY app/ /app/app/
COPY config/ /app/config/
COPY scripts/cache_models.py /app/scripts/cache_models.py
ENV PYTHONPATH=/app RUNTIME_TARGET=gpu OCR_DEVICE=gpu PRELOAD=false MODEL_DIR=/home/voight/.paddlex
RUN mkdir -p "$MODEL_DIR" && python scripts/cache_models.py

FROM cpu-assets AS cpu
COPY --chown=voight:voight app/ /app/app/
COPY --chown=voight:voight config/ /app/config/
USER voight
ENV RUNTIME_TARGET=cpu OCR_DEVICE=cpu PRELOAD=false MODEL_DIR=/home/voight/.paddlex
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/v1/health/ready').read()"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

FROM cpu AS cpu-test
USER root
COPY scripts/ /app/scripts/
COPY tests/ /app/tests/
RUN python -m unittest tests/test_config.py tests/test_contracts.py tests/test_batched_inference.py tests/test_v1_api.py -v

FROM gpu-assets AS gpu
COPY --chown=voight:voight app/ /app/app/
COPY --chown=voight:voight config/ /app/config/
USER voight
ENV RUNTIME_TARGET=gpu OCR_DEVICE=gpu PRELOAD=false MODEL_DIR=/home/voight/.paddlex
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/v1/health/ready').read()"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
